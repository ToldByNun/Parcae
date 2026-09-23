# Optional CPack helpers for deb/rpm when invoked from a CMake install tree.
# Prefer packaging/linux/build_packages.sh (fpm) for the release matrix filenames.

set(CPACK_PACKAGE_NAME "parcae")
set(CPACK_PACKAGE_VENDOR "Parcae")
set(CPACK_PACKAGE_CONTACT "noreply@example.com")
set(CPACK_PACKAGE_DESCRIPTION_SUMMARY "Parcae Liber Primus cryptanalysis toolkit")
set(CPACK_RESOURCE_FILE_LICENSE "${CMAKE_SOURCE_DIR}/LICENSE")
set(CPACK_RESOURCE_FILE_README "${CMAKE_SOURCE_DIR}/README.md")

if(NOT DEFINED CPACK_PACKAGE_VERSION)
  set(CPACK_PACKAGE_VERSION "${PROJECT_VERSION}")
endif()

set(CPACK_DEBIAN_PACKAGE_SECTION "science")
set(CPACK_DEBIAN_PACKAGE_DEPENDS "libc6, libstdc++6")
set(CPACK_DEBIAN_PACKAGE_RECOMMENDS "cmake, g++, python3")
set(CPACK_DEBIAN_FILE_NAME "DEB-DEFAULT")

include(CPack)
