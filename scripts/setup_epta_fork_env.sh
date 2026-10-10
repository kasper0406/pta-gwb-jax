#!/usr/bin/env bash
# Build the EPTA DR2 "fork oracle" environment (docs/M3B_PLAN.md Sec. 4.5, decision D6) WITHOUT root.
#
# It contains EPTA's modified enterprise and enterprise_extensions at pinned, source-audited
# revisions reproducing the chain-generating model (fork audit, docs/M3B0_VALIDATION.md Sec. 2), on top of the D1
# evaluator profile `published-tempo2-v1` (tempo2 2026.04.1, libstempo 2.5.1 rebuilt against it):
#
#   enterprise             gitlab.in2p3.fr/epta/enterprise            tag EPTADR2-v1.1 = 607c2853
#   enterprise_extensions  gitlab.in2p3.fr/epta/enterprise_extensions 051173f4 (2023-03-14)
#       (representative of the equivalence class {23c63a15, 051173f4}: after the orf_bins keyword,
#        before d3248419, which renamed the J1713 dip parameters; the released chains carry the
#        pre-rename names `J1713+0747_J1713+0747_dmexp_1_*`. EPTADR2-v1.1 =
#        7619622a differs from 051173f4 on the DR2new CRN/HD path only by that rename and an
#        unused pseed pass-through; see the audit)
#   PTMCMCSampler 2.1.1, numpy 1.26 (the forks predate numpy 2), scipy 1.11, scikit-sparse 0.4.12.
#
# It also creates the two runtime-source envs of the pinned tempo2 runtime reproducing the stored likelihoods
# (docs/M3B0_VALIDATION.md Sec. 3): conda-forge tempo2 2023.05.1 (build hcb8dc1c_5; its T2runtime
# is the base) and 2022.05.1 (build h1c8e422_2; only its clock/gps2utc.clk is used). Only their
# data files are used, never their binaries; scripts/m3b_epta_prepare.py assembles the runtime.
#
# The env is separate from both the project venv and the plain tempo2 oracle env
# (scripts/setup_tempo2_env.sh); run it through scripts/eptapy. Idempotent.
#
# Usage: scripts/setup_epta_fork_env.sh
#        EF_ENV=/other/prefix scripts/setup_epta_fork_env.sh
set -euo pipefail

MM_DIR="${MM_DIR:-$HOME/.local/opt/micromamba}"
EF_ENV="${EF_ENV:-$HOME/.local/opt/epta-fork-env}"
SRC="${EF_SRC:-$HOME/.local/opt/epta-fork-src}"
# tempo2 2026.04.1 is the D1 evaluator profile; other versions (EF_ENV + T2_VERSION) are
# diagnostics only (docs/M3B0_VALIDATION.md, fingerprint diagnosis)
T2_VERSION="${T2_VERSION:-2026.04.1}"
ENT_URL=https://gitlab.in2p3.fr/epta/enterprise.git
EXT_URL=https://gitlab.in2p3.fr/epta/enterprise_extensions.git
ENT_REV=607c28533acc18b4ed7a743ca568ffc7e41a9137   # tag EPTADR2-v1.1
EXT_REV=051173f46e64918aa4df9e9fea8fbc04a79839f9
LIBSTEMPO_SDIST="https://files.pythonhosted.org/packages/40/b4/e890d7d0f5c072f95fa7b36b9a66eaf00c59b7be6a0b1eea3f529cd14042/libstempo-2.5.1.tar.gz"

MM="$MM_DIR/bin/micromamba"
export MAMBA_ROOT_PREFIX="$MM_DIR/root"
PY="$EF_ENV/bin/python"
export TEMPO2="$EF_ENV/share/tempo2"
unset PYTHONPATH PYTHONHOME VIRTUAL_ENV
export PYTHONNOUSERSITE=1

if [[ ! -x "$PY" ]]; then
  "$MM" create -y -p "$EF_ENV" -c conda-forge --override-channels \
    python=3.11 "numpy=1.26" "scipy=1.11" "tempo2=$T2_VERSION" "scikit-sparse=0.4.12" \
    astropy ephem jplephem cython setuptools setuptools-scm wheel pip packaging matplotlib
fi

# conda-forge healpy builds either need numpy 2 or pull tempo2 back to 2024.12.1: use the PyPI wheel
"$PY" -m pip install --no-deps --no-cache-dir -q "healpy==1.16.6"

# libstempo rebuilt against the env's tempo2 (same reason as scripts/setup_tempo2_env.sh: ABI)
installed_t2="$(sed -n 's/^#define TEMPO2_h_VER "\(.*\)"/\1/p' "$EF_ENV/include/tempo2.h")"
[[ "$installed_t2" == "$T2_VERSION" ]] || { echo "tempo2 $installed_t2 != $T2_VERSION" >&2; exit 1; }
installed_t2="$("$PY" -c "from packaging import version; print(version.parse('$installed_t2'))")"
have_t2="$("$PY" -c "import libstempo.libstempo as L; print(L.tempo2version())" 2>/dev/null || echo none)"
if [[ "$have_t2" != "$installed_t2" ]]; then
  tmp="$(mktemp -d)"
  curl -sfL "$LIBSTEMPO_SDIST" | tar -xz -C "$tmp"
  ( cd "$tmp/libstempo-2.5.1" && TEMPO2_PREFIX="$EF_ENV" "$PY" -m pip install --no-deps --no-build-isolation --no-cache-dir --force-reinstall . )
  rm -rf "$tmp"
fi

mkdir -p "$SRC"
[[ -d "$SRC/enterprise" ]] || git clone -q "$ENT_URL" "$SRC/enterprise"
[[ -d "$SRC/enterprise_extensions" ]] || git clone -q "$EXT_URL" "$SRC/enterprise_extensions"
( cd "$SRC/enterprise" && git fetch -q --tags && git checkout -q "$ENT_REV" )
ext_full="$EXT_REV"; ( cd "$SRC/enterprise_extensions" && git fetch -q --tags )
( cd "$SRC/enterprise_extensions" && git checkout -q "$ext_full" )

# The forks' 2022/2023 setup.py files do not build under current setuptools, and enterprise's
# metadata says python<3.10; the pinned checkouts are put on sys.path with a .pth file instead
# (pure-python packages, no compiled parts).
site="$("$PY" -c 'import sysconfig; print(sysconfig.get_paths()["purelib"])')"
"$PY" -m pip uninstall -y -q enterprise-pulsar enterprise_extensions 2>/dev/null || true
printf '%s\n%s\n' "$SRC/enterprise" "$SRC/enterprise_extensions" > "$site/epta_fork.pth"
"$PY" -m pip install --no-deps --no-cache-dir -q "ptmcmcsampler==2.1.1" six numdifftools emcee "scikit-learn<1.4" joblib threadpoolctl "setuptools<70"  # enterprise 3.3 imports pkg_resources

"$PY" - <<EOF
import enterprise, enterprise_extensions, libstempo, libstempo.libstempo as L, numpy, scipy
from enterprise_extensions import models
import inspect
src = inspect.getsource(models.model_general)
assert "name='{0}_dmexp_{1}'.format(p.name,dd+1)" in src, "unexpected enterprise_extensions revision"
print("enterprise", enterprise.__file__)
print("enterprise_extensions", enterprise_extensions.__version__, enterprise_extensions.__file__)
print("libstempo", libstempo.__version__, "tempo2", L.tempo2version(), "numpy", numpy.__version__, "scipy", scipy.__version__)
EOF
for spec in "2023.05.1=hcb8dc1c_5" "2022.05.1=h1c8e422_2"; do
  v="${spec%%=*}"
  [[ -d "$HOME/.local/opt/t2rt-$v/share/tempo2" ]] || \
    "$MM" create -y -p "$HOME/.local/opt/t2rt-$v" -c conda-forge --override-channels "tempo2=$spec"
done
echo "ENT_REV=$ENT_REV"
echo "EXT_REV=$ext_full"
echo ">> EPTA fork env OK: $EF_ENV (use scripts/eptapy)"
