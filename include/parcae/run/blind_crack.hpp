#ifndef BLIND_CRACK_HPP
#define BLIND_CRACK_HPP

#include "parcae/batch/batch_execution.hpp"
#include "parcae/batch/batch_hit.hpp"
#include "parcae/batch/batch_result.hpp"
#include "parcae/batch/batch_runner.hpp"
#include "parcae/cli/console_dashboard.hpp"
#include "parcae/cli/console_progress_snapshot.hpp"
#include "parcae/core/index29.hpp"
#include "parcae/core/status.hpp"
#include "parcae/core/status_or.hpp"
#include "parcae/corpus/fixture.hpp"
#include "parcae/corpus/fixture_loader.hpp"
#include "parcae/corpus/token_stream.hpp"
#include "parcae/generate/affine_candidate_generator.hpp"
#include "parcae/generate/atbash_candidate_generator.hpp"
#include "parcae/generate/atbash_caesar_candidate_generator.hpp"
#include "parcae/generate/caesar_candidate_generator.hpp"
#include "parcae/generate/transform_candidate.hpp"
#include "parcae/gematria/gematria_profile.hpp"
#include "parcae/gematria/latin_codec.hpp"
#include "parcae/score/expected_frequency_table.hpp"
#include "parcae/score/score_request.hpp"
#include "parcae/tool/api.hpp"
#include "parcae/tool/context.hpp"
#include "parcae/transform/identity_transform.hpp"
#include "parcae/transform/transform_direction.hpp"
#include "parcae/transform/transform_id.hpp"
#include "parcae/validate/plaintext_normalizer.hpp"

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstddef>
#include <filesystem>
#include <iomanip>
#include <optional>
#include <span>
#include <sstream>
#include <string>
#include <utility>
#include <vector>

#include <nlohmann/json.hpp>

/// Blind crack attempt: ciphertext-only, additive family battery, χ² ranking.
///
/// There are no unsolved LP2 transcripts in-repo. Locked Tier-A fixtures are
/// used as stand-ins: plaintext is **not** used for scoring (oracle check only).
class BlindCrack {
public:
    struct Hit {
        std::string candidate_id;
        std::string transform_id;
        double score = 0.0;
        std::string latin_preview;
        std::vector<Index29> indices;
    };

    struct FixtureReport {
        std::string fixture_id;
        std::string true_transform_id;
        std::size_t rune_count = 0;
        std::size_t candidate_count = 0;
        double wall_seconds = 0.0;
        /// C × T / wall — decrypt+χ² work units per second.
        double runes_per_sec = 0.0;
        Hit top;
        bool cracked = false;
        bool family_in_battery = false;
    };

    struct Report {
        std::vector<FixtureReport> fixtures;
        double total_wall_seconds = 0.0;
        std::size_t cracked_count = 0;
        std::size_t attempted_count = 0;
        std::uint64_t total_runes_scored = 0;
        double aggregate_runes_per_sec = 0.0;
    };

    [[nodiscard]] static StatusOr<Report> run_all_locked(const Context& ctx) {
        const std::filesystem::path solved = ctx.data_root() / "fixtures" / "solved";
        if (!std::filesystem::is_directory(solved)) {
            return Status::error("BlindCrack: fixtures/solved missing");
        }

        StatusOr<ExpectedFrequencyTable> freqs = ctx.load_english_gp_expected();
        if (!freqs.ok()) {
            return freqs.status();
        }
        StatusOr<GematriaProfile> profile = ctx.load_gematria();
        if (!profile.ok()) {
            return profile.status();
        }
        const LatinCodec codec(profile.value());
        const PlaintextNormalizer normalizer(codec);

        Report report;
        const auto t_all0 = std::chrono::steady_clock::now();
        for (const std::filesystem::directory_entry& entry :
             std::filesystem::directory_iterator(solved)) {
            if (!entry.is_directory()) {
                continue;
            }
            StatusOr<Fixture> fixture = FixtureLoader::load_directory(entry.path().string());
            if (!fixture.ok() || fixture.value().verification_status() != "locked") {
                continue;
            }
            StatusOr<FixtureReport> one =
                crack_one(ctx, fixture.value(), freqs.value(), codec, normalizer);
            if (!one.ok()) {
                return one.status();
            }
            if (one.value().cracked) {
                ++report.cracked_count;
            }
            ++report.attempted_count;
            report.total_runes_scored += static_cast<std::uint64_t>(one.value().candidate_count) *
                                        static_cast<std::uint64_t>(one.value().rune_count);
            report.fixtures.push_back(std::move(one.value()));
        }
        const auto t_all1 = std::chrono::steady_clock::now();
        report.total_wall_seconds = std::chrono::duration<double>(t_all1 - t_all0).count();
        report.aggregate_runes_per_sec =
            report.total_wall_seconds > 0.0
                ? static_cast<double>(report.total_runes_scored) / report.total_wall_seconds
                : 0.0;
        return report;
    }

    [[nodiscard]] static std::string format_runes_per_sec(double rps) {
        return ConsoleDashboard::format_throughput(rps);
    }

    /// Human report: shared `ConsoleDashboard` panel + fixture table.
    [[nodiscard]] static std::string format(const Report& report) {
        const ConsoleProgressSnapshot snap = to_snapshot(report);
        std::ostringstream out;
        out << ConsoleDashboard::format_panel(snap, 28) << '\n';
        out << "(locked fixtures as stand-in; no unsolved LP2 transcripts in-repo)\n";
        out << "Battery: identity + atbash + caesar + atbash_caesar + affine (~871)\n";
        out << "Score:   chi2_english_gp_v0 - plaintext NOT used for ranking\n";
        out << "Throughput: C x T / wall (expand+score)\n\n";

        out << "Fixture              T    Cands     ms      runes/s  Top hit                         Cracked?\n";
        out << "----------------------------------------------------------------------------------------------\n";
        for (const FixtureReport& f : report.fixtures) {
            const double ms = f.wall_seconds * 1000.0;
            std::string top = f.top.candidate_id;
            if (top.size() > 30) {
                top = top.substr(0, 27) + "...";
            }
            out << std::left << std::setw(18) << f.fixture_id << "  " << std::right << std::setw(4)
                << f.rune_count << "  " << std::setw(6) << f.candidate_count << "  " << std::setw(6)
                << std::fixed << std::setprecision(1) << ms << "  " << std::setw(12)
                << format_runes_per_sec(f.runes_per_sec) << "  " << std::left << std::setw(30)
                << top << "  " << (f.cracked ? "YES" : "no");
            if (!f.family_in_battery) {
                out << "  (true=" << f.true_transform_id << " outside battery)";
            }
            out << '\n';
            if (!f.top.latin_preview.empty()) {
                out << "                     preview: " << f.top.latin_preview << '\n';
            }
        }
        out << "----------------------------------------------------------------------------------------------\n";
        out << "Cracked " << report.cracked_count << " / " << report.attempted_count << " in "
            << std::setprecision(3) << std::fixed << report.total_wall_seconds << " s total\n";
        out << "Aggregate throughput: " << format_runes_per_sec(report.aggregate_runes_per_sec)
            << "  (" << report.total_runes_scored << " rune-evals)\n";
        out << ConsoleDashboard::format_line(snap) << "  [done]\n";
        return out.str();
    }

private:
    BlindCrack() = delete;

    [[nodiscard]] static ConsoleProgressSnapshot to_snapshot(const Report& report) {
        ConsoleProgressSnapshot snap;
        snap.set_tool("blind-crack");
        snap.set_score_id("chi2_english_gp_v0");
        snap.set_backend("cpu");
        snap.set_stage("done");
        snap.set_candidates_done(report.attempted_count);
        snap.set_candidates_total(report.attempted_count);
        snap.set_elapsed_seconds(report.total_wall_seconds);
        snap.set_runes_per_sec(report.aggregate_runes_per_sec);
        if (report.total_wall_seconds > 0.0 && report.attempted_count > 0) {
            snap.set_candidates_per_sec(
                static_cast<double>(report.attempted_count) / report.total_wall_seconds);
        }
        // Prefer a cracked top hit; else best (lowest) χ² among fixtures.
        std::optional<double> best;
        std::string best_label;
        for (const FixtureReport& f : report.fixtures) {
            if (!best.has_value() || f.top.score < best.value()) {
                best = f.top.score;
                best_label = f.fixture_id;
                if (!f.top.candidate_id.empty()) {
                    best_label += " " + f.top.candidate_id;
                    if (best_label.size() > 40) {
                        best_label = best_label.substr(0, 37) + "...";
                    }
                }
            }
        }
        if (best.has_value()) {
            snap.set_best_score(best);
            snap.set_best_label(std::move(best_label));
        }
        return snap;
    }

    [[nodiscard]] static bool family_in_battery(const std::string& transform_id) {
        return transform_id == "identity" || transform_id == "atbash" ||
               transform_id == "caesar" || transform_id == "affine" ||
               transform_id == "compose";
    }

    [[nodiscard]] static StatusOr<std::vector<TransformCandidate>> expand_battery(
        std::span<const Index29> cipher) {
        std::vector<TransformCandidate> all;

        StatusOr<std::vector<Index29>> id_out = IdentityTransform{}.apply(
            cipher, nlohmann::json::object(), TransformDirection::Decrypt);
        if (!id_out.ok()) {
            return id_out.status();
        }
        all.emplace_back(
            "identity",
            TransformId::identity(),
            TransformDirection::Decrypt,
            nlohmann::json::object(),
            std::move(id_out.value()));

        auto append = [&](StatusOr<std::vector<TransformCandidate>> part) -> Status {
            if (!part.ok()) {
                return part.status();
            }
            for (TransformCandidate& c : part.value()) {
                all.push_back(std::move(c));
            }
            return Status::success();
        };

        Status s = append(AtbashCandidateGenerator::generate(cipher));
        if (!s.ok()) {
            return s;
        }
        s = append(CaesarCandidateGenerator::generate(cipher));
        if (!s.ok()) {
            return s;
        }
        s = append(AtbashCaesarCandidateGenerator::generate(cipher));
        if (!s.ok()) {
            return s;
        }
        s = append(AffineCandidateGenerator::generate(cipher));
        if (!s.ok()) {
            return s;
        }
        return all;
    }

    [[nodiscard]] static StatusOr<FixtureReport> crack_one(
        const Context& ctx,
        const Fixture& fixture,
        const ExpectedFrequencyTable& freqs,
        const LatinCodec& codec,
        const PlaintextNormalizer& normalizer) {
        StatusOr<TokenStream> stream = ToolApi::tokenize(ctx, fixture.ciphertext());
        if (!stream.ok()) {
            return stream.status();
        }
        const std::vector<Index29> cipher = stream.value().consumable_indices();
        if (cipher.empty()) {
            return Status::error("BlindCrack: empty consumable stream for " + fixture.id());
        }

        ScoreRequest request;
        request.expected_frequencies = &freqs;

        const auto t0 = std::chrono::steady_clock::now();
        StatusOr<std::vector<TransformCandidate>> candidates = expand_battery(cipher);
        if (!candidates.ok()) {
            return candidates.status();
        }
        StatusOr<BatchResult> ranked = BatchRunner::run(
            candidates.value(),
            "chi2_english_gp_v0",
            /*k=*/1,
            request,
            BatchExecution::Parallel);
        if (!ranked.ok()) {
            return ranked.status();
        }
        const auto t1 = std::chrono::steady_clock::now();

        FixtureReport report;
        report.fixture_id = fixture.id();
        report.true_transform_id = fixture.transform_id();
        report.family_in_battery = family_in_battery(fixture.transform_id());
        report.rune_count = cipher.size();
        report.candidate_count = candidates.value().size();
        report.wall_seconds = std::chrono::duration<double>(t1 - t0).count();
        const double work = static_cast<double>(report.candidate_count) *
                            static_cast<double>(report.rune_count);
        report.runes_per_sec =
            report.wall_seconds > 0.0 ? (work / report.wall_seconds) : 0.0;

        if (ranked.value().top().empty()) {
            return Status::error("BlindCrack: empty top-k for " + fixture.id());
        }
        const BatchHit& best = ranked.value().top()[0];
        const TransformCandidate& best_cand = candidates.value()[best.source_index()];
        report.top.candidate_id = best.candidate_id();
        report.top.transform_id = best_cand.transform_id().str();
        report.top.score = best.score();
        report.top.indices = best_cand.output_indices();

        const std::string latin = codec.latinize(report.top.indices);
        report.top.latin_preview = latin.substr(0, std::min<std::size_t>(72, latin.size()));

        StatusOr<std::string> expected = normalizer.normalize(fixture.plaintext());
        if (expected.ok()) {
            StatusOr<std::string> recovered_norm = normalizer.normalize(latin);
            if (recovered_norm.ok()) {
                report.cracked = (recovered_norm.value() == expected.value());
            }
        }
        return report;
    }
};

#endif  // BLIND_CRACK_HPP
