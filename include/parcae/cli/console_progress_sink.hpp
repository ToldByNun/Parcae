#ifndef CONSOLE_PROGRESS_SINK_HPP
#define CONSOLE_PROGRESS_SINK_HPP

#include "parcae/cli/console_progress_snapshot.hpp"

#include <string_view>

/// Observe-only progress hook for long-running CLI / library work.
///
/// Implementations MUST NOT affect ranking, digests, or return values.
/// Default no-op: `ConsoleProgressNoOpSink`.
class ConsoleProgressSink {
public:
    virtual ~ConsoleProgressSink() = default;

    /// Periodic or throttled tick during a stage (e.g. each scored candidate).
    virtual void on_progress(const ConsoleProgressSnapshot& snapshot) = 0;

    /// Stage transition (e.g. expand → score). Implementations SHOULD paint
    /// immediately even when progress ticks are throttled.
    virtual void on_stage(std::string_view stage, const ConsoleProgressSnapshot& snapshot) = 0;
};

/// Sink that discards all events (library default / agent quiet path).
class ConsoleProgressNoOpSink : public ConsoleProgressSink {
public:
    void on_progress(const ConsoleProgressSnapshot& /*snapshot*/) override {}

    void on_stage(std::string_view /*stage*/,
                  const ConsoleProgressSnapshot& /*snapshot*/) override {}
};

#endif // CONSOLE_PROGRESS_SINK_HPP
