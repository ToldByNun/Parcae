#!/usr/bin/env bash
# Derive Parcae SemVer X.Y.Z from CMakeLists.txt (project VERSION).
# Optional override: PARCAE_VERSION=1.2.3
set -euo pipefail

packaging_root="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd "${packaging_root}/.." && pwd)"

if [[ -n "${PARCAE_VERSION:-}" ]]; then
  echo "${PARCAE_VERSION}"
  exit 0
fi

cmake_lists="${repo_root}/CMakeLists.txt"
if [[ ! -f "${cmake_lists}" ]]; then
  echo "error: missing ${cmake_lists}" >&2
  exit 1
fi

# project(Parcae …) may be multi-line: VERSION on its own line.
version="$(
  tr '\n' ' ' < "${cmake_lists}" \
    | grep -oE 'project[[:space:]]*\([[:space:]]*Parcae[^)]*VERSION[[:space:]]+[0-9]+\.[0-9]+\.[0-9]+' \
    | head -n1 \
    | grep -oE '[0-9]+\.[0-9]+\.[0-9]+$' || true
)"

if [[ -z "${version}" ]]; then
  # Fallback: first bare VERSION x.y.z near top of file.
  version="$(
    grep -E '^[[:space:]]*VERSION[[:space:]]+[0-9]+\.[0-9]+\.[0-9]+' "${cmake_lists}" \
      | head -n1 \
      | grep -oE '[0-9]+\.[0-9]+\.[0-9]+' || true
  )"
fi

if [[ -z "${version}" ]]; then
  echo "error: could not parse project(Parcae VERSION x.y.z) from CMakeLists.txt" >&2
  exit 1
fi

echo "${version}"
