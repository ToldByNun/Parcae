#!/usr/bin/env bash
# Compile theories/examples/*.py via parcae-compile and reject the committed
# stale-major fixture under tests/fixtures/dsl/theories/.
#
# Usage:
#   scripts/check-dsl-examples.sh <parcae-compile> <parcae-validate> [workdir]
#
# Exit 0 only when portable example theories compile, ignore-without-allow is
# rejected, and the stale-major fixture fails validate (non-zero exit / all_ok false).

set -euo pipefail

if [[ $# -lt 2 ]]; then
  echo "usage: $0 <parcae-compile> <parcae-validate> [workdir]" >&2
  exit 2
fi

COMPILE="$1"
VALIDATE="$2"
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
EXAMPLES="${ROOT}/theories/examples"
FIXTURES_DATA="${ROOT}/tests/fixtures/dsl"

OWN_WORKDIR=0
if [[ $# -ge 3 ]]; then
  WORKDIR="$3"
else
  WORKDIR="$(mktemp -d "${TMPDIR:-/tmp}/parcae-dsl-examples.XXXXXX")"
  OWN_WORKDIR=1
fi

mkdir -p "${WORKDIR}"
cleanup() {
  if [[ "${OWN_WORKDIR}" -eq 1 ]]; then
    rm -rf "${WORKDIR}"
  fi
}
trap cleanup EXIT

echo "== compile new_math_example.py =="
"${COMPILE}" "${EXAMPLES}/new_math_example.py" --data-dir "${WORKDIR}"
echo "== compile full_lifecycle_example.py =="
"${COMPILE}" "${EXAMPLES}/full_lifecycle_example.py" --data-dir "${WORKDIR}"
echo "== compile matrix_builtins_example.py =="
"${COMPILE}" "${EXAMPLES}/matrix_builtins_example.py" --data-dir "${WORKDIR}"
echo "== compile param_select_example.py =="
"${COMPILE}" "${EXAMPLES}/param_select_example.py" --data-dir "${WORKDIR}"

echo "== ignore_divergent_example.py must fail without --allow-dsl-ignores =="
set +e
"${COMPILE}" "${EXAMPLES}/ignore_divergent_example.py" --data-dir "${WORKDIR}" \
  >/tmp/parcae-ignore-denied.out 2>/tmp/parcae-ignore-denied.err
ignore_rc=$?
set -e
if [[ "${ignore_rc}" -eq 0 ]]; then
  echo "expected non-zero exit when #ignore is used without --allow-dsl-ignores" >&2
  cat /tmp/parcae-ignore-denied.out >&2 || true
  cat /tmp/parcae-ignore-denied.err >&2 || true
  exit 1
fi
echo "ignore without allow rejected (exit ${ignore_rc}) — ok"

echo "== stale fixture must fail validate =="
set +e
"${VALIDATE}" --theory stale_major_demo@1 --data-dir "${FIXTURES_DATA}" --json \
  >/tmp/parcae-stale-validate.json 2>/tmp/parcae-stale-validate.err
stale_rc=$?
set -e
if [[ "${stale_rc}" -eq 0 ]]; then
  echo "expected non-zero exit for stale dsl_spec_version fixture" >&2
  cat /tmp/parcae-stale-validate.json >&2 || true
  cat /tmp/parcae-stale-validate.err >&2 || true
  exit 1
fi
echo "stale fixture rejected (exit ${stale_rc}) — ok"

echo "dsl examples + stale fixture check passed"
