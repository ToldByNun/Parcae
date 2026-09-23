#ifndef VERSION_HPP
#define VERSION_HPP

// Values are injected by CMake when building against parcae::core; the
// fallbacks keep the header usable for tooling that only parses includes.

#ifndef PARCAE_VERSION_MAJOR
#define PARCAE_VERSION_MAJOR 0
#endif

#ifndef PARCAE_VERSION_MINOR
#define PARCAE_VERSION_MINOR 7
#endif

#ifndef PARCAE_VERSION_PATCH
#define PARCAE_VERSION_PATCH 0
#endif

#ifndef PARCAE_VERSION_STRING
#define PARCAE_VERSION_STRING "0.7.0"
#endif

class Version {
public:
    static constexpr int major = PARCAE_VERSION_MAJOR;
    static constexpr int minor = PARCAE_VERSION_MINOR;
    static constexpr int patch = PARCAE_VERSION_PATCH;

private:
};

#endif // VERSION_HPP
