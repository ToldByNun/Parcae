# Optional CPack RPM helper. Prefer packaging/linux/build_packages.sh for exact filenames.

set(CPACK_PACKAGE_NAME "parcae")
set(CPACK_PACKAGE_VENDOR "Parcae")
set(CPACK_PACKAGE_DESCRIPTION_SUMMARY "Parcae Liber Primus cryptanalysis toolkit")
set(CPACK_RESOURCE_FILE_LICENSE "${CMAKE_SOURCE_DIR}/LICENSE")

if(NOT DEFINED CPACK_PACKAGE_VERSION)
  set(CPACK_PACKAGE_VERSION "${PROJECT_VERSION}")
endif()

set(CPACK_RPM_PACKAGE_LICENSE "MIT")
set(CPACK_RPM_PACKAGE_REQUIRES "glibc, libstdc++")
set(CPACK_RPM_FILE_NAME "RPM-DEFAULT")

include(CPack)
