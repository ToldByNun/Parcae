#!/usr/bin/env bash
# Create parcae-X.Y.Z-source.tar.gz via git archive.
# Usage: build_source.sh --out <dir> [--version x.y.z] [--repo <dir>]
set -euo pipefail

packaging_root="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd "${packaging_root}/.." && pwd)"
version="$("${packaging_root}/version.sh")"
out=""

while [[ $# -gt 0 ]]; do
  case "$1" in
    --out) out="$2"; shift 2 ;;
    --version) version="$2"; shift 2 ;;
    --repo) repo_root="$2"; shift 2 ;;
    *) echo "unknown: $1" >&2; exit 1 ;;
  esac
done

if [[ -z "${out}" ]]; then
  echo "usage: $0 --out <dir>" >&2
  exit 1
fi

mkdir -p "${out}"
name="parcae-${version}-source"
prefix="${name}/"
target="${out}/${name}.tar.gz"

git -C "${repo_root}" archive --format=tar.gz --prefix="${prefix}" -o "${target}" HEAD
echo "Wrote ${target}"
