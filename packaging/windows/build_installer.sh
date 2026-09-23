#!/usr/bin/env bash
# Build Windows installer(s) with Inno Setup after staging.
# Usage:
#   build_installer.sh --flavor cpu|cuda|full --stage <dir> --out <dir> [--version x.y.z]
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
  echo "usage: $0 --flavor cpu|cuda|full --stage <dir> --out <dir> [--version x.y.z]" >&2
  exit 1
fi

case "${flavor}" in
  cpu)
    suffix="-cpu"
    app_name="Parcae (CPU)"
    app_id="{{A7C3E5F1-9B2D-4E8A-B1C0-111111111111}"
    ;;
  cuda)
    suffix="-cuda"
    app_name="Parcae (CUDA)"
    app_id="{{A7C3E5F1-9B2D-4E8A-B1C0-222222222222}"
    ;;
  full)
    suffix=""
    app_name="Parcae"
    app_id="{{A7C3E5F1-9B2D-4E8A-B1C0-333333333333}"
    ;;
  *)
    echo "error: --flavor must be cpu|cuda|full" >&2
    exit 1
    ;;
esac

mkdir -p "${out}"
mkdir -p "${stage}/packaging/windows"
cp -a "${packaging_root}/windows/bootstrap.ps1" "${stage}/packaging/windows/bootstrap.ps1"

iscc_bin="${ISCC:-}"
if [[ -z "${iscc_bin}" ]]; then
  if command -v iscc >/dev/null 2>&1; then
    iscc_bin="$(command -v iscc)"
  elif [[ -f "/c/Users/${USER}/AppData/Local/Programs/Inno Setup 6/ISCC.exe" ]]; then
    iscc_bin="/c/Users/${USER}/AppData/Local/Programs/Inno Setup 6/ISCC.exe"
  elif [[ -f "/c/Program Files (x86)/Inno Setup 6/ISCC.exe" ]]; then
    iscc_bin="/c/Program Files (x86)/Inno Setup 6/ISCC.exe"
  elif [[ -f "C:/Program Files (x86)/Inno Setup 6/ISCC.exe" ]]; then
    iscc_bin="C:/Program Files (x86)/Inno Setup 6/ISCC.exe"
  else
    echo "error: Inno Setup ISCC.exe not found; install Inno Setup 6 or set ISCC=" >&2
    exit 1
  fi
fi

stage_win="${stage}"
out_win="${out}"
if command -v cygpath >/dev/null 2>&1; then
  stage_win="$(cygpath -w "$(cd "${stage}" && pwd)")"
  out_win="$(cygpath -w "$(mkdir -p "${out}" && cd "${out}" && pwd)")"
fi

# Quote defines that contain spaces for ISCC.
"${iscc_bin}" \
  "//DParcaeVersion=${version}" \
  "//DParcaeFlavor=${flavor}" \
  "//DParcaeSuffix=${suffix}" \
  "//DParcaeAppName=${app_name}" \
  "//DParcaeAppIdGuid=${app_id}" \
  "//DParcaeStage=${stage_win}" \
  "//DParcaeOut=${out_win}" \
  "${packaging_root}/windows/Parcae.iss"

echo "Built Windows installer flavor=${flavor} -> ${out}/Parcae-v${version}-windows-x64${suffix}.exe"
