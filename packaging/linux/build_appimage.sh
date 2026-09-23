#!/usr/bin/env bash
# Build a Parcae AppImage from a staged payload (requires linuxdeploy + appimagetool
# or continuous linuxdeploy AppImage which embeds both).
#
# Usage: build_appimage.sh --flavor cpu|cuda|full --stage <dir> --out <dir> [--version x.y.z]
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

case "${flavor}" in
  cpu) suffix="-cpu" ;;
  cuda) suffix="-cuda" ;;
  full) suffix="" ;;
  *) echo "bad flavor" >&2; exit 1 ;;
esac

mkdir -p "${out}"
workdir="$(mktemp -d)"
trap 'rm -rf "${workdir}"' EXIT

appdir="${workdir}/Parcae.AppDir"
mkdir -p "${appdir}/usr/bin" "${appdir}/usr/share/parcae" "${appdir}/usr/share/applications"

# Payload lives under usr/share/parcae; bin tools linked into usr/bin.
cp -a "${stage}/." "${appdir}/usr/share/parcae/"
# Prefer catalog as desktop entry binary; also expose dispatcher.
if [[ -x "${appdir}/usr/share/parcae/bin/parcae-catalog" ]]; then
  ln -sf ../share/parcae/bin/parcae-catalog "${appdir}/usr/bin/parcae-catalog"
fi
if [[ -x "${appdir}/usr/share/parcae/bin/parcae" ]]; then
  ln -sf ../share/parcae/bin/parcae "${appdir}/usr/bin/parcae"
fi
# Symlink all tools for convenience.
for t in "${appdir}/usr/share/parcae/bin"/parcae-*; do
  [[ -e "$t" ]] || continue
  base="$(basename "$t")"
  ln -sf "../share/parcae/bin/${base}" "${appdir}/usr/bin/${base}"
done

cat > "${appdir}/AppRun" <<'EOF'
#!/usr/bin/env bash
set -euo pipefail
HERE="$(dirname "$(readlink -f "$0")")"
export PARCAE_DATA_DIR="${PARCAE_DATA_DIR:-${HERE}/usr/share/parcae/data}"
export PATH="${HERE}/usr/bin:${PATH}"
if [[ $# -eq 0 ]]; then
  exec "${HERE}/usr/bin/parcae" --help
fi
# If first arg is a known tool suffix or full name, dispatch; else pass through.
exec "${HERE}/usr/bin/parcae" "$@"
EOF
chmod +x "${appdir}/AppRun"

cat > "${appdir}/usr/share/applications/parcae.desktop" <<EOF
[Desktop Entry]
Type=Application
Name=Parcae
Comment=Liber Primus cryptanalysis toolkit
Exec=parcae
Icon=parcae
Categories=Development;Science;
Terminal=true
EOF

# appimagetool expects .desktop + icon at AppDir root.
cp -a "${appdir}/usr/share/applications/parcae.desktop" "${appdir}/parcae.desktop"

# Minimal placeholder icon (AppImage tooling wants one).
mkdir -p "${appdir}/usr/share/icons/hicolor/256x256/apps"
printf '\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR\x00\x00\x00\x01\x00\x00\x00\x01\x08\x06\x00\x00\x00\x1f\x15\xc4\x89\x00\x00\x00\nIDATx\x9cc\x00\x01\x00\x00\x05\x00\x01\r\n-\xb4\x00\x00\x00\x00IEND\xaeB`\x82' \
  > "${appdir}/usr/share/icons/hicolor/256x256/apps/parcae.png"
cp -a "${appdir}/usr/share/icons/hicolor/256x256/apps/parcae.png" "${appdir}/parcae.png"

out_name="Parcae-v${version}-linux-x86_64${suffix}.AppImage"
export ARCH=x86_64
export VERSION="${version}"

if command -v appimagetool >/dev/null 2>&1; then
  appimagetool "${appdir}" "${out}/${out_name}"
elif [[ -n "${LINUXDEPLOY:-}" && -x "${LINUXDEPLOY}" ]]; then
  "${LINUXDEPLOY}" --appdir "${appdir}" --output appimage
  built="$(ls -1 "${workdir}"/*.AppImage 2>/dev/null | head -n1 || true)"
  if [[ -z "${built}" ]]; then
    built="$(ls -1 ./*.AppImage 2>/dev/null | head -n1 || true)"
  fi
  if [[ -z "${built}" ]]; then
    echo "error: linuxdeploy ran but no AppImage appeared" >&2
    exit 1
  fi
  mv "${built}" "${out}/${out_name}"
elif command -v linuxdeploy >/dev/null 2>&1; then
  linuxdeploy --appdir "${appdir}" --output appimage
  built="$(ls -1 ./*.AppImage 2>/dev/null | head -n1 || true)"
  mv "${built}" "${out}/${out_name}"
else
  echo "error: need appimagetool or linuxdeploy on PATH (or LINUXDEPLOY=...)" >&2
  exit 1
fi

chmod +x "${out}/${out_name}"
echo "Wrote ${out}/${out_name}"
