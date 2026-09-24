#ifndef BENCH_FORMATTER_HPP
#define BENCH_FORMATTER_HPP

#include "parcae/bench/bench_report.hpp"
#include "parcae/cli/console_dashboard.hpp"

#include <iomanip>
#include <sstream>
#include <string>
#include <string_view>

/// Human-readable bench tables (ConsoleDashboard throughput vocabulary).
class BenchFormatter {
public:
    /// Compact rate label for table columns (`15.00B`, `3.25M`, raw).
    [[nodiscard]] static std::string format_rps(double rps) {
        std::ostringstream out;
        out << std::fixed;
        if (rps >= 1.0e9) {
            out << std::setprecision(2) << (rps / 1.0e9) << "B";
        } else if (rps >= 1.0e6) {
            out << std::setprecision(2) << (rps / 1.0e6) << "M";
        } else if (rps >= 1.0e3) {
            out << std::setprecision(2) << (rps / 1.0e3) << "k";
        } else {
            out << std::setprecision(2) << rps;
        }
        return out.str();
    }

    /// Full dashboard label (`3.25B runes/s`).
    [[nodiscard]] static std::string format_throughput(double runes_per_sec) {
        return ConsoleDashboard::format_throughput(runes_per_sec);
    }

    [[nodiscard]] static std::string format_target(double target_min, double target_max) {
        std::ostringstream target;
        if (target_max > 0.0) {
            target << format_rps(target_min) << "-" << format_rps(target_max);
        } else {
            target << ">=" << format_rps(target_min);
        }
        return target.str();
    }

    /// Multi-section human report (SLO / accuracy / hardware / probe rows).
    [[nodiscard]] static std::string format(const BenchReport::Document& doc) {
        std::ostringstream out;
        out << "PARCAE — BENCH (" << BenchReport::suite_str(doc.suite()) << ")\n";
        out << "Toolkit: " << doc.toolkit_version() << '\n';
        out << "Metric: repeats x C x T / median-of-3 (setup excluded); "
               "keys/s = repeats x C / elapsed\n";
        out << "Peaks: practical ceilings (%peak must stay at or under 100)\n\n";

        emit_section(out, "SLO TIERS", doc, [](const BenchReport::Row& r) {
            return r.suite() == BenchReport::Suite::Slo &&
                   (r.name() == "T1" || r.name() == "T2" || r.name() == "T3");
        });
        emit_section(out, "TRANSFORM FAMILIES", doc, [](const BenchReport::Row& r) {
            return r.name().size() >= 2 && r.name()[0] == 'F' && r.name()[1] == '.';
        });
        emit_section(out, "COMPOSE", doc, [](const BenchReport::Row& r) {
            return r.name().size() >= 2 && r.name()[0] == 'C' && r.name()[1] == '.';
        });
        emit_section(out, "ACCURACY", doc, [](const BenchReport::Row& r) {
            return r.suite() == BenchReport::Suite::Accuracy;
        });
        emit_section(out, "HARDWARE", doc, [](const BenchReport::Row& r) {
            return r.suite() == BenchReport::Suite::Hardware;
        });
        emit_section(out, "PROBE", doc, [](const BenchReport::Row& r) {
            return r.suite() == BenchReport::Suite::Probe;
        });

        // Catch-all for rows that did not match a named section (e.g. suite=all extras).
        emit_section(out, "OTHER", doc, [&](const BenchReport::Row& r) {
            const bool slo_primary = r.suite() == BenchReport::Suite::Slo &&
                                     (r.name() == "T1" || r.name() == "T2" || r.name() == "T3");
            const bool family = r.name().size() >= 2 && r.name()[0] == 'F' && r.name()[1] == '.';
            const bool compose = r.name().size() >= 2 && r.name()[0] == 'C' && r.name()[1] == '.';
            const bool accuracy = r.suite() == BenchReport::Suite::Accuracy;
            const bool hardware = r.suite() == BenchReport::Suite::Hardware;
            const bool probe = r.suite() == BenchReport::Suite::Probe;
            return !(slo_primary || family || compose || accuracy || hardware || probe);
        });

        out << (doc.all_pass() ? "ALL ROWS PASS\n" : "ROWS FAILED\n");
        return out.str();
    }

private:
    BenchFormatter() = delete;

    template <typename Pred>
    static void emit_section(std::ostringstream& out, std::string_view title,
                             const BenchReport::Document& doc, Pred&& pred) {
        bool any = false;
        for (const BenchReport::Row& row : doc.rows()) {
            if (pred(row)) {
                any = true;
                break;
            }
        }
        if (!any) {
            return;
        }

        out << title << '\n';
        out << "Tier              Workload                      backend  runes/s     target      "
               "est.peak   %peak  result\n";
        out << "------------------------------------------------------------------------"
               "--------------------------------\n";
        for (const BenchReport::Row& row : doc.rows()) {
            if (!pred(row)) {
                continue;
            }
            const double pct = row.percent_peak();
            out << std::left << std::setw(17) << row.name() << " " << std::setw(27)
                << row.workload() << " " << std::setw(7) << BenchReport::backend_str(row.backend())
                << " " << std::right << std::setw(10) << format_rps(row.runes_per_sec()) << "  "
                << std::left << std::setw(11) << format_target(row.target_min(), row.target_max())
                << "  " << std::right << std::setw(8) << format_rps(row.estimated_peak()) << "  "
                << std::setw(5) << std::fixed << std::setprecision(0) << pct << "%" << "  "
                << BenchReport::status_str(row.status()) << '\n';
            out << "      C=" << row.candidates() << " T=" << row.tokens()
                << " reps=" << row.repeats();
            if (!row.detail().empty()) {
                out << "  (" << row.detail() << ")";
            }
            out << '\n';
        }
        out << "------------------------------------------------------------------------"
               "--------------------------------\n\n";
    }
};

#endif // BENCH_FORMATTER_HPP
