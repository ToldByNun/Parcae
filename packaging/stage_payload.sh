#!/usr/bin/env bash
# Stage a release payload directory for installer / tarball / AppImage / deb / rpm.
#
# Usage:
#   stage_payload.sh --out <dir> --flavor cpu|cuda|full \
#     [--bin-cpu <dir>] [--bin-cuda <dir>] [--repo <dir>]
#
# Layout written to --out:
#   bin/          CPU CLIs (or only CUDA CLIs when flavor=cuda)
#   bin-cuda/     CUDA CLIs (full flavor only)
#   data/         committed data tree
#   src/          full source snapshot (no build trees)
#   python/       DSL stubs package
#   agents/       optional agent package
#   LICENSE README.md VERSION
#   scripts/install-python-editable.sh|.ps1
set -euo pipefail

packaging_root="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd "${packaging_root}/.." && pwd)"
# shellcheck source=version.sh
version="$("${packaging_root}/version.sh")"

out=""
flavor=""
bin_cpu=""
bin_cuda=""

usage() {
  echo "usage: $0 --out <dir> --flavor cpu|cuda|full [--bin-cpu <dir>] [--bin-cuda <dir>] [--repo <dir>]" >&2
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --out) out="$2"; shift 2 ;;
    --flavor) flavor="$2"; shift 2 ;;
    --bin-cpu) bin_cpu="$2"; shift 2 ;;
    --bin-cuda) bin_cuda="$2"; shift 2 ;;
    --repo) repo_root="$2"; shift 2 ;;
    -h|--help) usage; exit 0 ;;
    *) echo "unknown arg: $1" >&2; usage; exit 1 ;;
  esac
done

if [[ -z "${out}" || -z "${flavor}" ]]; then
  usage
  exit 1
fi

case "${flavor}" in
  cpu|cuda|full) ;;
  *) echo "error: --flavor must be cpu|cuda|full" >&2; exit 1 ;;
esac

if [[ "${flavor}" == "cpu" || "${flavor}" == "full" ]]; then
  if [[ -z "${bin_cpu}" ]]; then
    echo "error: --bin-cpu required for flavor=${flavor}" >&2
    exit 1
  fi
fi
if [[ "${flavor}" == "cuda" || "${flavor}" == "full" ]]; then
  if [[ -z "${bin_cuda}" ]]; then
    echo "error: --bin-cuda required for flavor=${flavor}" >&2
    exit 1
  fi
fi

rm -rf "${out}"
mkdir -p "${out}"

copy_bins() {
  local src="$1"
  local dst="$2"
  mkdir -p "${dst}"
  # Accept either a tools output dir or a flat directory of parcae-* binaries.
  if compgen -G "${src}/parcae-*" > /dev/null; then
    # shellcheck disable=SC2086
    cp -a ${src}/parcae-* "${dst}/"
  elif [[ -d "${src}" ]]; then
    # Multi-config layout: Release/ or Debug/
    if compgen -G "${src}/Release/parcae-*" > /dev/null; then
      cp -a "${src}/Release"/parcae-* "${dst}/"
    elif compgen -G "${src}/parcae-*" > /dev/null; then
      cp -a "${src}"/parcae-* "${dst}/"
    else
      echo "error: no parcae-* binaries under ${src}" >&2
      exit 1
    fi
  else
    echo "error: bin dir not found: ${src}" >&2
    exit 1
  fi
  # Drop test helpers from release payloads if present.
  rm -f "${dst}/parcae-parity-gen" "${dst}/parcae-parity-gen.exe" || true
  chmod +x "${dst}"/parcae-* 2>/dev/null || true
}

case "${flavor}" in
  cpu)
    copy_bins "${bin_cpu}" "${out}/bin"
    ;;
  cuda)
    copy_bins "${bin_cuda}" "${out}/bin"
    ;;
  full)
    copy_bins "${bin_cpu}" "${out}/bin"
    copy_bins "${bin_cuda}" "${out}/bin-cuda"
    ;;
esac

# data/ — committed fixtures/profiles (skip local scratch via rsync filters if available)
mkdir -p "${out}/data"
if command -v rsync >/dev/null 2>&1; then
  rsync -a \
    --exclude 'workspaces/*' --include 'workspaces/_example/***' \
    --exclude 'theories/*' --include 'theories/README.md' \
    "${repo_root}/data/" "${out}/data/"
else
  cp -a "${repo_root}/data/." "${out}/data/"
  # Best-effort cleanup of local scratch mirrors.
  find "${out}/data/workspaces" -mindepth 1 -maxdepth 1 ! -name '_example' -exec rm -rf {} + 2>/dev/null || true
  find "${out}/data/theories" -mindepth 1 -maxdepth 1 ! -name 'README.md' -exec rm -rf {} + 2>/dev/null || true
fi

# Source snapshot for developers (exclude build trees and VCS).
mkdir -p "${out}/src"
if command -v git >/dev/null 2>&1 && git -C "${repo_root}" rev-parse --is-inside-work-tree >/dev/null 2>&1; then
  git -C "${repo_root}" archive --format=tar HEAD | tar -x -C "${out}/src"
else
  if command -v rsync >/dev/null 2>&1; then
    rsync -a \
      --exclude '.git' --exclude 'build' --exclude 'build-*' --exclude 'out' \
      --exclude '.cache' --exclude 'release' --exclude '*.egg-info' \
      --exclude '__pycache__' --exclude '.venv' \
      "${repo_root}/" "${out}/src/"
  else
    echo "error: need git or rsync to stage sources" >&2
    exit 1
  fi
fi

cp -a "${repo_root}/python" "${out}/python"
cp -a "${repo_root}/agents" "${out}/agents"
# Drop local venv / caches from staged packages.
rm -rf "${out}/python/.venv" "${out}/python/.pytest_cache" "${out}/python"/*.egg-info \
  "${out}/agents/.venv" "${out}/agents/.pytest_cache" "${out}/agents"/*.egg-info 2>/dev/null || true

cp -a "${repo_root}/LICENSE" "${out}/LICENSE"
cp -a "${repo_root}/README.md" "${out}/README.md"
printf '%s\n' "${version}" > "${out}/VERSION"
printf '%s\n' "${flavor}" > "${out}/FLAVOR"

mkdir -p "${out}/scripts"
cat > "${out}/scripts/install-python-editable.sh" <<'EOF'
#!/usr/bin/env bash
set -euo pipefail
root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
python3 -m pip install -e "${root}/python/."
python3 -m pip install -e "${root}/agents/."
echo "Installed parcae DSL stubs and agents editable packages."
EOF
chmod +x "${out}/scripts/install-python-editable.sh"

cat > "${out}/scripts/install-python-editable.ps1" <<'EOF'
$ErrorActionPreference = "Stop"
$root = Split-Path -Parent $PSScriptRoot
python -m pip install -e (Join-Path $root "python")
python -m pip install -e (Join-Path $root "agents")
Write-Host "Installed parcae DSL stubs and agents editable packages."
EOF

# Tiny dispatcher for AppImage / developer shell.
cat > "${out}/bin/parcae" <<'EOF'
#!/usr/bin/env bash
# List or forward to installed parcae-* CLIs next to this script.
set -euo pipefail
here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
if [[ $# -lt 1 || "$1" == "-h" || "$1" == "--help" ]]; then
  echo "Parcae toolkit — available commands:"
  ls -1 "${here}"/parcae-* 2>/dev/null | xargs -n1 basename | sed 's/^/  /'
  echo "Usage: parcae <tool-suffix> [args...]   e.g. parcae catalog --json"
  exit 0
fi
cmd="$1"; shift
target="${here}/parcae-${cmd}"
if [[ ! -x "${target}" && ! -f "${target}.exe" ]]; then
  echo "error: unknown tool '${cmd}' (expected ${target})" >&2
  exit 1
fi
if [[ -f "${target}.exe" ]]; then
  exec "${target}.exe" "$@"
fi
exec "${target}" "$@"
EOF
chmod +x "${out}/bin/parcae"

echo "Staged flavor=${flavor} version=${version} -> ${out}"
