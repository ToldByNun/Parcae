#!/usr/bin/env bash
# Regenerate CPU parity goldens into a temp dir and byte-compare against data/parity/.
# Usage: check-parity-goldens.sh <parcae-parity-gen> <data-dir> [committed-parity-dir]
set -euo pipefail

if [[ $# -lt 2 || $# -gt 3 ]]; then
  echo "Usage: $0 <parcae-parity-gen> <data-dir> [committed-parity-dir]" >&2
  exit 2
fi

GEN=$1
DATA=$2
COMMITTED=${3:-"${DATA}/parity"}

if [[ ! -x "${GEN}" && ! -f "${GEN}" ]]; then
  echo "parity-gen not found: ${GEN}" >&2
  exit 1
fi
if [[ ! -d "${COMMITTED}" ]]; then
  echo "committed parity dir missing: ${COMMITTED}" >&2
  exit 1
fi

TMP=$(mktemp -d)
trap 'rm -rf "${TMP}"' EXIT

"${GEN}" --data-dir "${DATA}" --out "${TMP}"

fail=0
while IFS= read -r -d '' f; do
  rel=${f#"${TMP}/"}
  committed="${COMMITTED}/${rel}"
  if [[ ! -f "${committed}" ]]; then
    echo "MISSING in committed tree: ${rel}" >&2
    fail=1
    continue
  fi
  if ! cmp -s "${f}" "${committed}"; then
    echo "DRIFT: ${rel}" >&2
    fail=1
  fi
done < <(find "${TMP}" -type f -print0 | sort -z)

# Committed extras that the generator should own (ignore README and similar docs).
while IFS= read -r -d '' f; do
  rel=${f#"${COMMITTED}/"}
  case "${rel}" in
    README.md|*.md) continue ;;
  esac
  if [[ ! -f "${TMP}/${rel}" ]]; then
    echo "EXTRA committed file (not regenerated): ${rel}" >&2
    fail=1
  fi
done < <(find "${COMMITTED}" -type f -print0 | sort -z)

if [[ "${fail}" -ne 0 ]]; then
  echo "Parity goldens out of date. Regenerate with parcae-parity-gen and commit." >&2
  exit 1
fi

echo "Parity goldens match (${COMMITTED})."
