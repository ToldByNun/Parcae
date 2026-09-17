#ifndef BATCH_EXECUTION_HPP
#define BATCH_EXECUTION_HPP

#include <string_view>

/// How `BatchRunner` computes per-candidate scores before deterministic reduction.
enum class BatchExecution {
    /// Score candidates in input order on the calling thread (source of truth).
    Serial,
    /// Score with a fixed chunked thread pool; reduction stays serial + ordered.
    /// See `BatchOrdering` / docs/spec/scores.md § Batch parallelism.
    Parallel,
};

class BatchExecutionUtil {
public:
    [[nodiscard]] static constexpr std::string_view to_string(BatchExecution exec) noexcept {
        switch (exec) {
        case BatchExecution::Serial:
            return "serial";
        case BatchExecution::Parallel:
            return "parallel";
        }
        return "serial";
    }
};

#endif // BATCH_EXECUTION_HPP
