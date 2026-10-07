"""XLA:CPU YNNPACK-fusion miscompilation (jaxlib 0.11.2; docs/PERF.md, bench/xla_ynn_repro.py).

* ``ptagwb`` sets ``--xla_cpu_experimental_ynn_fusion_type=`` in XLA_FLAGS at import (appending,
  never clobbering); the compiled CPU HLO of the known-bad pattern must contain no ``__ynn_fusion``.
* Compiled reducer VJPs (production and fast) on XLA:CPU must equal the eager gradients for
  scalar-loss, constant nonzero, explicit-zero, partial (HD-like) and runtime cotangents with the
  workaround on. With it off (opt-out env var) the production reducer is known to fail for
  constant cotangents: documented in KNOWN_BAD_WITHOUT_WORKAROUND (not an xfail, which strict
  mode would turn into a failure); every other case must still pass.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

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


# Documented known failure without the workaround (an expected-failure test written as a passing
# one, because PTAGWB_REQUIRE_ORACLES / --require-oracles turns every xfail into a failure): all
# cases outside KNOWN_BAD must still pass; KNOWN_BAD may fail (measured 0.14-2.6 scaled error on the
# synthetic PTA, 4e273 on 67 NG15 pulsars) or pass (e.g. after an upstream XLA fix).
KNOWN_BAD_WITHOUT_WORKAROUND = {"prod/const"}


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
