#!/usr/bin/env bash
# Build .deb and/or .rpm from a staged payload using fpm (preferred) or dpkg-deb / rpmbuild fallbacks.
#
# Usage:
#   build_packages.sh --flavor cpu|cuda|full --stage <dir> --out <dir> [--formats deb,rpm]
set -euo pipefail

packaging_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
version="$("${packaging_root}/version.sh")"
flavor=""
stage=""
out=""
formats="deb,rpm"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --flavor) flavor="$2"; shift 2 ;;
    --stage) stage="$2"; shift 2 ;;
    --out) out="$2"; shift 2 ;;
    --version) version="$2"; shift 2 ;;
    --formats) formats="$2"; shift 2 ;;
    *) echo "unknown: $1" >&2; exit 1 ;;
  esac
done

if [[ -z "${flavor}" || -z "${stage}" || -z "${out}" ]]; then
  echo "usage: $0 --flavor cpu|cuda|full --stage <dir> --out <dir> [--formats deb,rpm]" >&2
  exit 1
fi

mkdir -p "${out}"

case "${flavor}" in
  cpu)
    deb_name="parcae_${version}_amd64_cpu.deb"
    rpm_name="parcae-${version}-1.x86_64.cpu.rpm"
    pkg_suffix=" (CPU)"
    depends_deb="libc6, libstdc++6"
    depends_rpm="glibc, libstdc++"
    ;;
  cuda)
    deb_name="parcae_${version}_amd64_cuda.deb"
    rpm_name="parcae-${version}-1.x86_64.cuda.rpm"
    pkg_suffix=" (CUDA)"
    depends_deb="libc6, libstdc++6"
    depends_rpm="glibc, libstdc++"
    ;;
  full)
    deb_name="parcae_${version}_amd64.deb"
    rpm_name="parcae-${version}-1.x86_64.rpm"
    pkg_suffix=""
    depends_deb="libc6, libstdc++6"
    depends_rpm="glibc, libstdc++"
    ;;
  *) echo "bad flavor" >&2; exit 1 ;;
esac

# Recommend toolchain packages in description; declare soft Recommends when using fpm.
description="Parcae Liber Primus cryptanalysis toolkit${pkg_suffix}. Includes CLIs, data/, and sources. Build extras: cmake (>= 3.25), g++ (C++20), python3 (>= 3.11)."

stage_pkg="$(mktemp -d)"
trap 'rm -rf "${stage_pkg}"' EXIT
mkdir -p "${stage_pkg}/opt/parcae" "${stage_pkg}/usr/bin"
cp -a "${stage}/." "${stage_pkg}/opt/parcae/"
for t in "${stage_pkg}/opt/parcae/bin"/parcae*; do
  [[ -e "$t" ]] || continue
  base="$(basename "$t")"
  ln -sf "/opt/parcae/bin/${base}" "${stage_pkg}/usr/bin/${base}"
done

IFS=',' read -r -a fmt_arr <<< "${formats}"

build_with_fpm() {
  local fmt="$1"
  local target="$2"
  local extra=()
  if [[ "${fmt}" == "deb" ]]; then
    extra+=(--depends "libc6" --depends "libstdc++6" --deb-recommends "cmake" --deb-recommends "g++" --deb-recommends "python3")
    if [[ "${flavor}" == "cuda" || "${flavor}" == "full" ]]; then
      extra+=(--deb-recommends "nvidia-cuda-toolkit")
    fi
  else
    extra+=(--depends "glibc" --depends "libstdc++")
  fi
  fpm -s dir -t "${fmt}" -n parcae -v "${version}" \
    --iteration 1 \
    --architecture x86_64 \
    --description "${description}" \
    --license MIT \
    --url "https://github.com/ToldByNun/Parcae" \
    --prefix / \
    "${extra[@]}" \
    -C "${stage_pkg}" \
    -p "${target}" \
    opt usr
}

for fmt in "${fmt_arr[@]}"; do
  fmt="$(echo "${fmt}" | tr -d '[:space:]')"
  case "${fmt}" in
    deb)
      target="${out}/${deb_name}"
      if command -v fpm >/dev/null 2>&1; then
        build_with_fpm deb "${target}"
      else
        # Minimal dpkg-deb fallback.
        debroot="$(mktemp -d)"
        mkdir -p "${debroot}/DEBIAN"
        cp -a "${stage_pkg}/." "${debroot}/"
        cat > "${debroot}/DEBIAN/control" <<EOF
Package: parcae
Version: ${version}
Section: science
Priority: optional
Architecture: amd64
Depends: ${depends_deb}
Recommends: cmake, g++, python3
Maintainer: Parcae <noreply@example.com>
Description: ${description}
EOF
        dpkg-deb --build "${debroot}" "${target}"
        rm -rf "${debroot}"
      fi
      echo "Wrote ${target}"
      ;;
    rpm)
      target="${out}/${rpm_name}"
      if command -v fpm >/dev/null 2>&1; then
        build_with_fpm rpm "${target}"
      else
        echo "error: rpm packaging requires fpm (gem install fpm) when rpmbuild recipe is not used" >&2
        exit 1
      fi
      echo "Wrote ${target}"
      ;;
    *)
      echo "unknown format: ${fmt}" >&2
      exit 1
      ;;
  esac
done
