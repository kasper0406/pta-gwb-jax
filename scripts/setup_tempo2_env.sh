#!/usr/bin/env bash
# Build an isolated tempo2 + libstempo "oracle" environment WITHOUT root, separate from the
# project's uv venv. Used only as an external reference implementation (see scripts/t2py).
#
# What it does (each step is skipped if already done, so the script is idempotent):
#   1. Downloads a static micromamba binary into $MM_DIR (default ~/.local/opt/micromamba).
#   2. Creates a conda-forge env at $T2_ENV (default ~/.local/opt/tempo2-env) with tempo2
#      (pinned, newest on conda-forge as of 2026-10), python 3.12, numpy, astropy, and the
#      runtime deps of libstempo, plus the build tools needed for step 3.
#   3. Rebuilds libstempo from the PyPI sdist against the env's tempo2 headers.
#      WHY: the conda-forge libstempo binary (2.5.1 build hf9745cd_1) was compiled against
#      tempo2 2025.02.1 but only pins `tempo2 >=2024.04.1`. tempo2 2026.04.1 changed the ABI
#      (3 new entries in the `param_label` enum + 2 new doubles in `struct pulsar`), so the
#      prebuilt extension would index the wrong parameters/fields. We detect the mismatch via
#      libstempo.libstempo.tempo2version() (compile-time TEMPO2_h_VER) vs the installed header.
#   4. Overlays NANOGrav observatory clock files (GBT/AO/VLA) from ipta/pulsar-clock-corrections (pinned commit) where
#      the upstream file extends LATER than the conda-bundled one (only files that already exist
#      locally; no new files are added, so tempo2's clock-chain resolution is unchanged).
#      The bundled gbt2gps.clk ends at MJD 58904 and ao2gps.clk at 58835, which makes tempo2
#      warn [CLK3] "no clock corrections available for clock UTC(gbt)" on NANOGrav 15-yr data.
#      Originals are kept in $TEMPO2/clock/.orig-conda/. Disable with T2_UPDATE_CLOCKS=0.
#   5. Ensures ephemeris/DE440.1950.2050 is present (it ships with the conda package; download
#      fallback from the tempo2 bitbucket repo otherwise) and runs a smoke test.
#
# Usage: scripts/setup_tempo2_env.sh
#        T2_ENV=/other/prefix MM_DIR=/other/mm scripts/setup_tempo2_env.sh
# Afterwards: scripts/t2py -c "import libstempo; print(libstempo.__version__)"
set -euo pipefail

MM_DIR="${MM_DIR:-$HOME/.local/opt/micromamba}"
T2_ENV="${T2_ENV:-$HOME/.local/opt/tempo2-env}"
TEMPO2_VERSION="${TEMPO2_VERSION:-2026.04.1}"
LIBSTEMPO_VERSION="${LIBSTEMPO_VERSION:-2.5.1}"
LIBSTEMPO_SDIST="https://files.pythonhosted.org/packages/40/b4/e890d7d0f5c072f95fa7b36b9a66eaf00c59b7be6a0b1eea3f529cd14042/libstempo-2.5.1.tar.gz"
CLOCK_REPO_SHA="${CLOCK_REPO_SHA:-15ce77d5a8c6bcfe0de2db95a5ce4d1027134b80}"  # ipta/pulsar-clock-corrections main @ 2026-10-08
T2_UPDATE_CLOCKS="${T2_UPDATE_CLOCKS:-1}"

MM="$MM_DIR/bin/micromamba"
export MAMBA_ROOT_PREFIX="$MM_DIR/root"
PY="$T2_ENV/bin/python"
export TEMPO2="$T2_ENV/share/tempo2"
unset PYTHONPATH PYTHONHOME VIRTUAL_ENV
export PYTHONNOUSERSITE=1

# --- 1. micromamba --------------------------------------------------------------------------
if [[ ! -x "$MM" ]]; then
  echo ">> downloading micromamba into $MM_DIR"
  mkdir -p "$MM_DIR"
  curl -Ls https://micro.mamba.pm/api/micromamba/linux-64/latest | tar -xj -C "$MM_DIR" bin/micromamba
fi
"$MM" --version

# --- 2. conda-forge env ---------------------------------------------------------------------
if [[ ! -x "$T2_ENV/bin/tempo2" || ! -x "$PY" ]]; then
  echo ">> creating env $T2_ENV"
  # libstempo is installed from conda-forge only to pull in its runtime deps (scipy, ephem,
  # pyerfa, matplotlib, ...); its binary is replaced in step 3.
  "$MM" create -y -p "$T2_ENV" -c conda-forge --override-channels \
    python=3.12 "tempo2=$TEMPO2_VERSION" "libstempo=$LIBSTEMPO_VERSION" numpy astropy \
    cython setuptools setuptools-scm wheel pip
fi
installed_t2="$(sed -n 's/^#define TEMPO2_h_VER "\(.*\)"/\1/p' "$T2_ENV/include/tempo2.h")"
echo ">> tempo2 header version: $installed_t2"

# --- 3. libstempo rebuilt against the installed tempo2 ---------------------------------------
want_t2="$("$PY" -c "from packaging import version; print(version.parse('$installed_t2'))")"
have_t2="$("$PY" -c "import libstempo.libstempo as L; print(L.tempo2version())" 2>/dev/null || echo none)"
if [[ "$have_t2" != "$want_t2" ]]; then
  echo ">> libstempo compiled against tempo2 '$have_t2' != installed '$want_t2'; rebuilding"
  if [[ -n "$(ls "$T2_ENV"/conda-meta/libstempo-*.json 2>/dev/null)" ]]; then
    "$MM" remove -y -p "$T2_ENV" --force libstempo   # --force: keep its deps
  fi
  tmp="$(mktemp -d)"
  curl -sfL "$LIBSTEMPO_SDIST" | tar -xz -C "$tmp"
  ( cd "$tmp/libstempo-$LIBSTEMPO_VERSION" && \
    TEMPO2_PREFIX="$T2_ENV" CC="${CC:-gcc}" CXX="${CXX:-g++}" \
      "$PY" -m pip install --no-deps --no-build-isolation --no-cache-dir --force-reinstall . )
  rm -rf "$tmp"
fi

# --- 4. clock-file overlay ------------------------------------------------------------------
if [[ "$T2_UPDATE_CLOCKS" == "1" ]]; then
  "$PY" - "$TEMPO2/clock" "$CLOCK_REPO_SHA" <<'EOF'
import os, shutil, sys, urllib.request
clk, sha = sys.argv[1], sys.argv[2]
base = f"https://raw.githubusercontent.com/ipta/pulsar-clock-corrections/{sha}/T2runtime/clock/"
bak = os.path.join(clk, ".orig-conda"); os.makedirs(bak, exist_ok=True)
def last_mjd(text):
    m = None
    for l in text.splitlines():
        p = l.split()
        if len(p) >= 2 and not l.startswith("#"):
            try: m = float(p[0]); float(p[1])
            except ValueError: pass
    return m
# Deliberately narrow: only the NANOGrav observatory files (stale in the conda bundle).
# PPTA (pks*) and MPTA (mk2utc) files in the bundle already cover DR3 / MPTA 4.5-yr spans.
for name in ["gbt2gps.clk", "ao2gps.clk", "vla2gps.clk"]:
    local = os.path.join(clk, name)
    if not os.path.exists(local):
        continue  # never add files: that could change which clock chain tempo2 picks
    try:
        up = urllib.request.urlopen(base + name, timeout=60).read().decode()
    except Exception as e:
        print(f"   clock {name}: upstream fetch failed ({e}); keeping local"); continue
    lo = open(local).read()
    if up == lo: continue
    a, b = last_mjd(lo), last_mjd(up)
    if a is not None and b is not None and b > a:
        if not os.path.exists(os.path.join(bak, name)): shutil.copy2(local, os.path.join(bak, name))
        os.unlink(local)  # env files are hardlinks into the micromamba pkg cache: never write in place
        open(local, "w").write(up)
        print(f"   clock {name}: updated, coverage end MJD {a} -> {b}")
EOF
fi

# --- 5. ephemeris + smoke test --------------------------------------------------------------
de440="$TEMPO2/ephemeris/DE440.1950.2050"
if [[ ! -s "$de440" ]]; then
  echo ">> DE440 missing, downloading"
  curl -sfL https://bitbucket.org/psrsoft/tempo2/raw/master/T2runtime/ephemeris/DE440.1950.2050 -o "$de440"
fi
for y in 2019 2020 2021 2022 2023; do
  [[ -s "$TEMPO2/clock/tai2tt_bipm$y.clk" ]] || echo "WARNING: tai2tt_bipm$y.clk missing" >&2
done
echo ">> tempo2 -v: $("$T2_ENV/bin/tempo2" -v 2>&1 | tail -1)"
"$PY" -c "import libstempo, libstempo.libstempo as L, numpy, astropy, os
print('libstempo', libstempo.__version__, 'built vs tempo2', L.tempo2version(), '| numpy', numpy.__version__, '| astropy', astropy.__version__)
print('TEMPO2 =', os.environ['TEMPO2'])"
echo ">> tempo2 oracle env OK: $T2_ENV  (use scripts/t2py)"
