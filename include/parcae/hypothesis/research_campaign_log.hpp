#ifndef RESEARCH_CAMPAIGN_LOG_HPP
#define RESEARCH_CAMPAIGN_LOG_HPP

#include "parcae/core/status.hpp"
#include "parcae/core/status_or.hpp"
#include "parcae/hypothesis/workspace_paths.hpp"
#include "parcae/tool/tool_backend.hpp"

#include <chrono>
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <sstream>
#include <string>
#include <string_view>
#include <utility>

/// Append-only research campaign audit log under `research/run.log`.
///
/// Ops guard for Liber Primus workspaces: record that work ran through Parcae
/// tools (not a Python Z29/χ² reimplementation). Paths in the log are
/// **workspace-relative** only (`hypothesis-workspace.md` § Determinism).
///
/// `run.log` is intentionally non-golden (wall clock / throughput); digests MUST
/// NOT hash this file.
class ResearchCampaignLog {
public:
    /// One search_cycle append. `runes_work = candidates_expanded * rune_count`
    /// when expanded is known; else `candidates_retained * rune_count` (lower bound).
    struct SearchCycleEntry {
        std::string tool{"search_cycle"};
        std::string family;
        std::string score_id;
        Backend backend_requested = Backend::Cpu;
        Backend export_backend = Backend::Cpu;
        std::size_t candidates_retained = 0;
        std::size_t candidates_expanded = 0; ///< 0 ⇒ unknown; log uses retained for work
        std::size_t rune_count = 0;
        double wall_seconds = 0.0;
        std::string batch_id;
        std::string created_utc;
        std::string notes; ///< short free text; MUST NOT contain host-absolute paths
    };

    /// True when `data/workspaces/<id>/research/` exists (research layout).
    [[nodiscard]] static bool research_layout_present(const std::filesystem::path& data_root,
                                                      std::string_view workspace_id) {
        StatusOr<std::filesystem::path> dir = WorkspacePaths::research_dir(data_root, workspace_id);
        if (!dir.ok()) {
            return false;
        }
        std::error_code ec;
        return std::filesystem::is_directory(dir.value(), ec) && !ec;
    }

    /// Append one line to `research/run.log` when the research dir exists.
    /// No-op success if research layout is absent (slim workspaces).
    [[nodiscard]] static Status append_search_cycle(const std::filesystem::path& data_root,
                                                    std::string_view workspace_id,
                                                    const SearchCycleEntry& entry) {
        if (!research_layout_present(data_root, workspace_id)) {
            return Status::success();
        }
        StatusOr<std::filesystem::path> dir = WorkspacePaths::research_dir(data_root, workspace_id);
        if (!dir.ok()) {
            return dir.status();
        }
        const std::filesystem::path path = dir.value() / "run.log";
        std::ofstream out(path, std::ios::binary | std::ios::app);
        if (!out) {
            return Status::error("ResearchCampaignLog: failed to open research/run.log");
        }

        const std::size_t expanded =
            entry.candidates_expanded > 0 ? entry.candidates_expanded : entry.candidates_retained;
        const std::uint64_t runes_work =
            static_cast<std::uint64_t>(expanded) * static_cast<std::uint64_t>(entry.rune_count);
        const double runes_per_s =
            (entry.wall_seconds > 0.0 && runes_work > 0)
                ? (static_cast<double>(runes_work) / entry.wall_seconds)
                : 0.0;

        std::ostringstream line;
        line << "event=search_cycle";
        line << " tool=" << sanitize_token(entry.tool);
        line << " family=" << sanitize_token(entry.family);
        line << " score_id=" << sanitize_token(entry.score_id);
        line << " backend_requested=" << BackendUtil::to_string(entry.backend_requested);
        line << " export_backend=" << BackendUtil::to_string(entry.export_backend);
        line << " candidates_retained=" << entry.candidates_retained;
        line << " candidates_expanded=" << expanded;
        line << " rune_count=" << entry.rune_count;
        line << " runes_work=" << runes_work;
        line << " wall_seconds=" << entry.wall_seconds;
        line << " runes_per_s=" << runes_per_s;
        if (!entry.batch_id.empty()) {
            line << " batch_id=" << sanitize_token(entry.batch_id);
        }
        if (!entry.created_utc.empty()) {
            line << " created_utc=" << sanitize_token(entry.created_utc);
        }
        if (!entry.notes.empty()) {
            line << " notes=" << sanitize_notes(entry.notes);
        }
        line << " engine=parcae"; // campaign guard marker — not a Python Z29 mirror
        line << '\n';

        out << line.str();
        if (!out) {
            return Status::error("ResearchCampaignLog: failed while writing research/run.log");
        }
        return Status::success();
    }

    /// Reject strings that look like host-absolute paths (Windows drive / UNC / Unix root).
    [[nodiscard]] static bool looks_host_absolute(std::string_view text) noexcept {
        if (text.empty()) {
            return false;
        }
        if (text[0] == '/' || text[0] == '\\') {
            return true;
        }
        if (text.size() >= 2 && text[1] == ':' &&
            ((text[0] >= 'A' && text[0] <= 'Z') || (text[0] >= 'a' && text[0] <= 'z'))) {
            return true;
        }
        if (text.size() >= 2 && text[0] == '\\' && text[1] == '\\') {
            return true;
        }
        return false;
    }

private:
    ResearchCampaignLog() = delete;

    [[nodiscard]] static std::string sanitize_token(std::string_view raw) {
        std::string out;
        out.reserve(raw.size());
        for (char c : raw) {
            if (c == ' ' || c == '\t' || c == '\n' || c == '\r') {
                out.push_back('_');
            } else {
                out.push_back(c);
            }
        }
        if (looks_host_absolute(out)) {
            return "REDACTED_ABS_PATH";
        }
        return out;
    }

    [[nodiscard]] static std::string sanitize_notes(std::string_view raw) {
        if (looks_host_absolute(raw)) {
            return "REDACTED_ABS_PATH";
        }
        return sanitize_token(raw);
    }
};

#endif // RESEARCH_CAMPAIGN_LOG_HPP
