#pragma once

// Project version macros. Prefer these over scattering string literals.
// Values are injected by CMake when building against parcae::core; the
// fallbacks keep the header usable for tooling that only parses includes.

#ifndef PARCAE_VERSION_MAJOR
#define PARCAE_VERSION_MAJOR 0
#endif

#ifndef PARCAE_VERSION_MINOR
#define PARCAE_VERSION_MINOR 1
#endif

#ifndef PARCAE_VERSION_PATCH
#define PARCAE_VERSION_PATCH 0
#endif

#ifndef PARCAE_VERSION_STRING
#define PARCAE_VERSION_STRING "0.1.0"
#endif

namespace parcae::core {

inline constexpr int version_major = PARCAE_VERSION_MAJOR;
inline constexpr int version_minor = PARCAE_VERSION_MINOR;
inline constexpr int version_patch = PARCAE_VERSION_PATCH;

}  // namespace parcae::core
