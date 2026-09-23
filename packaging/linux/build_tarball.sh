#!/usr/bin/env bash
# Build Linux tarball from a staged payload.
# Usage: build_tarball.sh --flavor cpu|cuda|full --stage <dir> --out <dir> [--version x.y.z]
set -euo pipefail

packaging_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
version="$("${packaging_root}/version.sh")"
flavor=""
stage=""
out=""

while [[ $# -gt 0 ]]; do
  case "$1" in
    --flavor) flavor="$2"; shift 2 ;;
    --stage) stage="$2"; shift 2 ;;
    --out) out="$2"; shift 2 ;;
    --version) version="$2"; shift 2 ;;
    *) echo "unknown: $1" >&2; exit 1 ;;
  esac
done

if [[ -z "${flavor}" || -z "${stage}" || -z "${out}" ]]; then
  echo "usage: $0 --flavor cpu|cuda|full --stage <dir> --out <dir>" >&2
  exit 1
fi

mkdir -p "${out}"
case "${flavor}" in
  cpu) suffix="-cpu" ;;
  cuda) suffix="-cuda" ;;
  full) suffix="" ;;
  *) echo "bad flavor" >&2; exit 1 ;;
esac

name="Parcae-v${version}-linux-x86_64${suffix}"
tmp="$(mktemp -d)"
trap 'rm -rf "${tmp}"' EXIT
mkdir -p "${tmp}/${name}"
cp -a "${stage}/." "${tmp}/${name}/"
rm -f "${out}/${name}.tar.gz"
tar -C "${tmp}" -czf "${out}/${name}.tar.gz" "${name}"
echo "Wrote ${out}/${name}.tar.gz"
