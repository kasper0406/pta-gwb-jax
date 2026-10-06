#!/usr/bin/env bash
# Install the optional "oracle" dependency group (enterprise-pulsar, enterprise_extensions,
# discovery) on a machine WITHOUT root access.
#
# enterprise imports scikit-sparse (sksparse.cholmod) unconditionally, and scikit-sparse
# ships only as an sdist that needs the SuiteSparse/CHOLMOD headers + shared libraries.
# Instead of `sudo apt install libsuitesparse-dev`, we `apt-get download` the Ubuntu 24.04
# packages, unpack them into a user prefix with `dpkg -x`, and build scikit-sparse against
# that prefix with an RPATH (DT_RPATH, so that libcholmod's own deps -- BLAS/LAPACK/metis --
# are found transitively).
#
# Usage: scripts/setup_oracle_env.sh            (idempotent)
#        SS_PREFIX=/some/dir scripts/setup_oracle_env.sh
set -euo pipefail

SS_PREFIX="${SS_PREFIX:-$HOME/.local/opt/suitesparse-7.6.1}"
LIB="$SS_PREFIX/usr/lib/x86_64-linux-gnu"
PKGS=(libsuitesparse-dev libcholmod5 libamd3 libcamd3 libcolamd3 libccolamd3
      libsuitesparseconfig7 libblas3 liblapack3 libgfortran5 libmetis5)

if [[ ! -e "$LIB/libcholmod.so.5" ]]; then
  tmp="$(mktemp -d)"
  ( cd "$tmp" && apt-get download "${PKGS[@]}" )
  mkdir -p "$SS_PREFIX"
  for d in "$tmp"/*.deb; do dpkg -x "$d" "$SS_PREFIX"; done
  rm -rf "$tmp"
fi
ln -sf blas/libblas.so.3 "$LIB/libblas.so.3"
ln -sf lapack/liblapack.so.3 "$LIB/liblapack.so.3"

export CFLAGS="-I$SS_PREFIX/usr/include/suitesparse ${CFLAGS:-}"
export LDFLAGS="-L$LIB -Wl,--disable-new-dtags,-rpath,$LIB ${LDFLAGS:-}"
export SUITESPARSE_INCLUDE_DIR="$SS_PREFIX/usr/include/suitesparse"
export SUITESPARSE_LIBRARY_DIR="$LIB"

cd "$(dirname "$0")/.."
uv sync --group oracle --reinstall-package scikit-sparse
uv run --group oracle python -c "import sksparse.cholmod, enterprise, enterprise_extensions, discovery; print('oracle env OK')"
