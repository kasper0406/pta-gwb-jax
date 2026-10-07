"""XLA:CPU YNNPACK-fusion miscompilation (jaxlib 0.11.2; docs/PERF.md, bench/xla_ynn_repro.py).

* ``ptagwb`` sets ``--xla_cpu_experimental_ynn_fusion_type=`` in XLA_FLAGS at import (appending,
  never clobbering); the compiled CPU HLO of the known-bad pattern must contain no ``__ynn_fusion``.
* Compiled reducer VJPs (production and fast) on XLA:CPU must equal the eager gradients for
  scalar-loss, constant nonzero, explicit-zero, partial (HD-like) and runtime cotangents with the
  workaround on, with it off (opt-out env var; both backward rules carry optimization barriers),
  and for four import orders in fresh processes (incl. JAX initialised before ``import ptagwb``,
  which must warn, or raise with PTAGWB_STRICT_XLA_FLAGS=1).
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

TESTS = Path(__file__).resolve().parent

PROBE = """
import os, json, jax
import ptagwb
from ptagwb.config import xla_cpu_ynn_fusion_active
print("RESULT " + json.dumps({"active": xla_cpu_ynn_fusion_active(), "flags": os.environ.get("XLA_FLAGS", ""),
                              "n_cpu": len(jax.devices("cpu"))}))
"""

CASES = """
import sys, json
sys.path.insert(0, {tests!r})
import ptagwb
import jax
from ptagwb.config import xla_cpu_ynn_fusion_active
from ptagwb.likelihood import PTALikelihood, precompute
from synthetic import make_pta, tspan
import ynn_cases
psrs, nd = make_pta(5, seed=0, n_epochs=70, signal=3e-7)
T = tspan(psrs)
like = PTALikelihood(precompute(psrs, nd, T, n_modes=30), T, n_modes=30, n_common=30, orf="curn", common="freespec")
print("RESULT " + json.dumps({{"backend": jax.default_backend(), "active": xla_cpu_ynn_fusion_active(),
                              "errors": ynn_cases.run_cases(like)}}))
"""


def _run(code, **env_extra):
    env = {k: v for k, v in os.environ.items() if k not in ("XLA_FLAGS", "PTAGWB_KEEP_XLA_CPU_YNN_FUSION")}
    env.update(JAX_PLATFORMS="cpu", **env_extra)
    r = subprocess.run([sys.executable, "-c", code], env=env, capture_output=True, text=True, timeout=1800, check=False)
    assert r.returncode == 0, r.stderr[-3000:]
    line = next(ln for ln in r.stdout.splitlines() if ln.startswith("RESULT "))
    return json.loads(line[len("RESULT "):])


def test_workaround_flag_is_effective():
    out = _run(PROBE)
    assert "--xla_cpu_experimental_ynn_fusion_type=" in out["flags"]
    assert out["active"] is False


def test_workaround_appends_to_existing_xla_flags():
    out = _run(PROBE, XLA_FLAGS="--xla_force_host_platform_device_count=3")
    assert out["flags"].startswith("--xla_force_host_platform_device_count=3")
    assert out["n_cpu"] == 3 and out["active"] is False


def test_opt_out_keeps_fusion():
    out = _run(PROBE, PTAGWB_KEEP_XLA_CPU_YNN_FUSION="1")
    assert out["active"] is True  # if this fails, XLA no longer forms the fusion: re-check the workaround


def _assert_cases(out, tol=1e-9):
    bad = {k: v for k, v in out["errors"].items() if not v <= tol}
    assert not bad, bad


def test_reducer_vjp_cpu_with_workaround():
    out = _run(CASES.format(tests=str(TESTS)))
    assert out["backend"] == "cpu" and out["active"] is False
    _assert_cases(out)


# Without the XLA flag (opt-out) both reducers must still be correct: their backward rules keep
# the vulnerable products behind optimization barriers. Before the production barrier was added,
# prod/const failed here (0.14-2.6 scaled error on the synthetic PTA, 4e273 on 67 NG15 pulsars).
KNOWN_BAD_WITHOUT_WORKAROUND: set[str] = set()  # was {"prod/const"} before the production barrier


def test_reducer_vjp_cpu_without_workaround_known_failures():
    out = _run(CASES.format(tests=str(TESTS)), PTAGWB_KEEP_XLA_CPU_YNN_FUSION="1")
    assert out["active"] is True
    _assert_cases({"errors": {k: v for k, v in out["errors"].items() if k not in KNOWN_BAD_WITHOUT_WORKAROUND}})
    print("known-bad cases without the workaround:", {k: out["errors"][k] for k in KNOWN_BAD_WITHOUT_WORKAROUND})


def test_fast_reducer_vjp_cpu_without_workaround():
    """Second defence: the fast reducer's optimization barriers keep it correct even with the
    YNNPACK fusion enabled."""
    out = _run(CASES.format(tests=str(TESTS)), PTAGWB_KEEP_XLA_CPU_YNN_FUSION="1")
    _assert_cases({"errors": {k: v for k, v in out["errors"].items() if k.startswith("hh/")}})


def test_reducer_vjp_default_backend():
    import jax  # noqa: F401
    sys.path.insert(0, str(TESTS))
    import ynn_cases
    from synthetic import make_pta, tspan

    from ptagwb.likelihood import PTALikelihood, precompute

    psrs, nd = make_pta(5, seed=0, n_epochs=70, signal=3e-7)
    T = tspan(psrs)
    like = PTALikelihood(precompute(psrs, nd, T, n_modes=30), T, n_modes=30, n_common=30, orf="curn", common="freespec")
    _assert_cases({"errors": ynn_cases.run_cases(like)})


ORDERS = {
    "ptagwb_first": "",
    "jax_imported_first": "import jax",
    "array_created_first": "import jax.numpy as jnp; jnp.ones(1).block_until_ready()",
    "devices_queried_first": "import jax; jax.devices()",
}

ORDER_CASES = """
import warnings
warnings.simplefilter("always")
{pre}
with warnings.catch_warnings(record=True) as w:
    import ptagwb
late_warning = any("initialised before `import ptagwb`" in str(x.message) for x in w)
import sys, json
sys.path.insert(0, {tests!r})
import jax
from ptagwb.config import xla_cpu_ynn_fusion_active
from ptagwb.likelihood import PTALikelihood, precompute
from synthetic import make_pta, tspan
import ynn_cases
psrs, nd = make_pta(5, seed=0, n_epochs=70, signal=3e-7)
T = tspan(psrs)
like = PTALikelihood(precompute(psrs, nd, T, n_modes=30), T, n_modes=30, n_common=30, orf="curn", common="freespec")
print("RESULT " + json.dumps({{"active": xla_cpu_ynn_fusion_active(), "late_warning": late_warning,
                              "errors": ynn_cases.run_cases(like)}}))
"""


@pytest.mark.parametrize("order", list(ORDERS))
def test_import_order(order):
    """Fresh process per import order: both reducers must give correct compiled gradients for
    nonzero-constant and explicit-zero (and all other) cotangents, whether or not the XLA flag
    took effect (the backward rules carry optimization barriers); a late import must warn."""
    out = _run(ORDER_CASES.format(pre=ORDERS[order], tests=str(TESTS)))
    late = order in ("array_created_first", "devices_queried_first")
    assert out["active"] is late
    assert out["late_warning"] is late
    for red in ("prod", "hh"):
        for case in ("const", "zeros"):
            assert f"{red}/{case}" in out["errors"]
    _assert_cases(out)


def test_late_import_strict_mode_raises():
    env = {k: v for k, v in os.environ.items() if k != "XLA_FLAGS"}
    env.update(JAX_PLATFORMS="cpu", PTAGWB_STRICT_XLA_FLAGS="1")
    r = subprocess.run([sys.executable, "-c", "import jax; jax.devices(); import ptagwb"], env=env,
                       capture_output=True, text=True, timeout=600, check=False)
    assert r.returncode != 0 and "XlaFlagsTooLateError" in r.stderr
