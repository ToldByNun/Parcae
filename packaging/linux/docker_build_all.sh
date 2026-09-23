#!/usr/bin/env bash
# Build all Linux release packages inside this container.
# Expects repo mounted at /src (or run from a writable copy) and output at /out.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${ROOT}"
# Capture SemVer before any `source /etc/os-release` (which exports VERSION=).
PARCAE_VER="$(packaging/version.sh)"
export PARCAE_VERSION="${PARCAE_VER}"
mkdir -p /out /work

export DEBIAN_FRONTEND=noninteractive
apt-get update -qq
apt-get install -y -qq \
  build-essential ninja-build git curl ca-certificates \
  libfuse2 rpm ruby ruby-dev file binutils gpg wget \
  >/dev/null

# CMake ≥ 3.25 (Ubuntu 22.04 apt is too old)
if ! cmake --version 2>/dev/null | grep -qE 'version 3\.(2[5-9]|[3-9][0-9])|version [4-9]'; then
  wget -qO- https://apt.kitware.com/keys/kitware-archive-latest.asc \
    | gpg --dearmor - > /usr/share/keyrings/kitware-archive-keyring.gpg
  # shellcheck disable=SC1091
  source /etc/os-release
  echo "deb [signed-by=/usr/share/keyrings/kitware-archive-keyring.gpg] https://apt.kitware.com/ubuntu/ ${VERSION_CODENAME} main" \
    > /etc/apt/sources.list.d/kitware.list
  apt-get update -qq
  apt-get install -y -qq cmake >/dev/null
fi
# Restore toolkit version after os-release clobber.
PARCAE_VER="$(packaging/version.sh)"
export PARCAE_VERSION="${PARCAE_VER}"
echo "Parcae version=${PARCAE_VER}"
cmake --version | head -n1

# Prefer GCC ≥ 12 (GCC 11 struggles with nested Options defaults).
if command -v g++-13 >/dev/null 2>&1; then
  export CC=gcc-13 CXX=g++-13
elif command -v g++-12 >/dev/null 2>&1; then
  export CC=gcc-12 CXX=g++-12
else
  apt-get install -y -qq g++-12 >/dev/null || true
  if command -v g++-12 >/dev/null 2>&1; then
    export CC=gcc-12 CXX=g++-12
  fi
fi
echo "Using CXX=${CXX:-g++}"
"${CXX:-g++}" --version | head -n1

gem install --no-document fpm >/dev/null

curl -fsSL -o /tmp/appimagetool \
  https://github.com/AppImage/appimagetool/releases/download/continuous/appimagetool-x86_64.AppImage
chmod +x /tmp/appimagetool
(cd /tmp && ./appimagetool --appimage-extract >/dev/null)
export PATH="/tmp/squashfs-root/usr/bin:${PATH}"
export APPIMAGE_EXTRACT_AND_RUN=1

chmod +x packaging/*.sh packaging/linux/*.sh 2>/dev/null || true

package_stage() {
  local flavor="$1"
  local stage="$2"
  packaging/linux/build_tarball.sh --flavor "${flavor}" --stage "${stage}" --out /out --version "${PARCAE_VER}"
  packaging/linux/build_packages.sh --flavor "${flavor}" --stage "${stage}" --out /out --version "${PARCAE_VER}" --formats deb,rpm
  # AppImage is best-effort (FUSE/desktop quirks); do not block cuda/full matrix.
  if ! packaging/linux/build_appimage.sh --flavor "${flavor}" --stage "${stage}" --out /out --version "${PARCAE_VER}"; then
    echo "WARNING: AppImage build failed for flavor=${flavor}; continuing" >&2
  fi
}

echo "==> Building Linux CPU"
cmake -S "${ROOT}" -B /work/build-cpu -G Ninja \
  -DCMAKE_BUILD_TYPE=Release \
  -DPARCAE_BUILD_TESTS=OFF \
  -DPARCAE_BUILD_TOOLS=ON \
  -DPARCAE_BUILD_CUDA=OFF
cmake --build /work/build-cpu -j"$(nproc)"
packaging/stage_payload.sh --out /work/stage-cpu --flavor cpu --bin-cpu /work/build-cpu/tools
package_stage cpu /work/stage-cpu

if [[ "${PARCAE_LINUX_CUDA:-1}" == "1" ]]; then
  echo "==> Building Linux CUDA"
  cmake -S "${ROOT}" -B /work/build-cuda -G Ninja \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_CUDA_HOST_COMPILER="${CXX:-g++}" \
    -DCMAKE_CUDA_ARCHITECTURES="75;86;89" \
    -DPARCAE_BUILD_TESTS=OFF \
    -DPARCAE_BUILD_TOOLS=ON \
    -DPARCAE_BUILD_CUDA=ON
  cmake --build /work/build-cuda -j"$(nproc)"
  packaging/stage_payload.sh --out /work/stage-cuda --flavor cuda --bin-cuda /work/build-cuda/tools
  package_stage cuda /work/stage-cuda

  echo "==> Staging Linux full (cpu+cuda bins)"
  packaging/stage_payload.sh --out /work/stage-full --flavor full \
    --bin-cpu /work/build-cpu/tools --bin-cuda /work/build-cuda/tools
  package_stage full /work/stage-full
else
  echo "Skipping Linux CUDA/full (PARCAE_LINUX_CUDA=0)"
fi

echo "==> Linux packages:"
ls -lah /out
