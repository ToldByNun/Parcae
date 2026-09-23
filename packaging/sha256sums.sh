#!/usr/bin/env bash
# Write SHA256SUMS for release artifacts in a directory.
# Usage: sha256sums.sh <release-dir>
set -euo pipefail

if [[ $# -ne 1 ]]; then
  echo "usage: $0 <release-dir>" >&2
  exit 1
fi

release_dir="$(cd "$1" && pwd)"
out="${release_dir}/SHA256SUMS"
rm -f "${out}"

(
  cd "${release_dir}"
  # Only checksum distributable artifacts (not README / .gitignore / notes).
  mapfile -t files < <(
    find . -maxdepth 1 -type f \( \
      -name '*.exe' -o -name '*.tar.gz' -o -name '*.AppImage' -o \
      -name '*.deb' -o -name '*.rpm' \
    \) | sed 's|^\./||' | sort
  )
  if [[ ${#files[@]} -eq 0 ]]; then
    echo "error: no release artifacts found in ${release_dir}" >&2
    exit 1
  fi
  if command -v sha256sum >/dev/null 2>&1; then
    sha256sum -- "${files[@]}"
  else
    shasum -a 256 -- "${files[@]}"
  fi
) > "${out}"

echo "Wrote ${out}"
