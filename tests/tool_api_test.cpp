#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>
#include <parcae/batch/batch_execution.hpp>
#include <parcae/batch/batch_runner.hpp>
#include <parcae/cli/console_progress_sink.hpp>
#include <parcae/gematria/rune_codec.hpp>
#include <parcae/tool/api.hpp>
#include <parcae/tool/context.hpp>
#include <parcae/tool/generate_candidates.hpp>
#include <parcae/tool/rank_candidates.hpp>
#include <parcae/tool/tool_backend.hpp>
#include <parcae/tool/transform_envelope.hpp>
#include <parcae/transform/caesar_transform.hpp>
#include <parcae/transform/transform_direction.hpp>

#if defined(PARCAE_HAS_CUDA)
#include "cuda_score.hpp"
#endif

#include <algorithm>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <mutex>
#include <optional>
#include <sstream>
#include <string>
#include <vector>

#ifndef PARCAE_TEST_DATA_DIR
#error "PARCAE_TEST_DATA_DIR must be defined"
#endif

namespace {

[[nodiscard]] Context test_ctx() {
    return Context{std::filesystem::path(PARCAE_TEST_DATA_DIR)};
}

[[nodiscard]] Index29 I(std::uint8_t v) {
    return Index29{v};
}

} // namespace

TEST_CASE("tool::tokenize round-trips a-warning ciphertext", "[tool][tokenize]") {
    const auto ctx = test_ctx();
    StatusOr<std::string> ciphertext = [&]() -> StatusOr<std::string> {
        const auto path = ctx.data_root() / "fixtures" / "solved" / "a-warning" / "ciphertext.txt";
        std::ifstream in(path, std::ios::binary);
        if (!in) {
            return Status::error("missing ciphertext");
        }
        std::ostringstream buf;
        buf << in.rdbuf();
        return buf.str();
    }();
    REQUIRE(ciphertext.ok());

    StatusOr<TokenStream> stream =
        ToolApi::tokenize(ctx, ciphertext.value(), "rtkd-separator-grammar-v0", true);
    REQUIRE(stream.ok());
    REQUIRE(stream.value().consumable_count() > 0);
    REQUIRE(stream.value().text() == ciphertext.value());
}

TEST_CASE("tool::apply_to_indices and apply_and_rebuild_text", "[tool][apply]") {
    const auto ctx = test_ctx();

    StatusOr<TransformEnvelope> env = TransformEnvelope::from_json(nlohmann::json{
        {"transform_id", "caesar"},
        {"direction", "encrypt"},
        {"params", {{"shift", 3}}},
    });
    REQUIRE(env.ok());

    const std::vector<Index29> plain = {I(0), I(1), I(2)};
    StatusOr<std::vector<Index29>> cipher = ToolApi::apply_to_indices(plain, env.value());
    REQUIRE(cipher.ok());
    REQUIRE(cipher.value() == std::vector<Index29>{I(3), I(4), I(5)});

    StatusOr<TransformEnvelope> decrypt = TransformEnvelope::from_json(nlohmann::json{
        {"transform_id", "caesar"},
        {"direction", "decrypt"},
        {"params", {{"shift", 3}}},
    });
    REQUIRE(decrypt.ok());
    StatusOr<std::vector<Index29>> recovered =
        ToolApi::apply_to_indices(cipher.value(), decrypt.value());
    REQUIRE(recovered.ok());
    REQUIRE(recovered.value() == plain);

    // Rebuild preserves separators in a tokenized page fragment.
    StatusOr<GematriaProfile> profile = ctx.load_gematria();
    REQUIRE(profile.ok());
    StatusOr<std::string> r0 = RuneCodec{profile.value()}.encode(I(0));
    StatusOr<std::string> r1 = RuneCodec{profile.value()}.encode(I(1));
    REQUIRE(r0.ok());
    REQUIRE(r1.ok());
    const std::string mixed = r0.value() + "-" + r1.value();
    StatusOr<TokenStream> stream = ToolApi::tokenize(ctx, mixed);
    REQUIRE(stream.ok());

    StatusOr<std::string> rebuilt =
        ToolApi::apply_and_rebuild_text(ctx, stream.value(), env.value());
    REQUIRE(rebuilt.ok());
    StatusOr<std::string> e3 = RuneCodec{profile.value()}.encode(I(3));
    StatusOr<std::string> e4 = RuneCodec{profile.value()}.encode(I(4));
    REQUIRE(e3.ok());
    REQUIRE(e4.ok());
    REQUIRE(rebuilt.value() == e3.value() + "-" + e4.value());
}

TEST_CASE("tool::apply_to_indices Backend::Cuda matches CPU when available", "[tool][backend]") {
    StatusOr<Backend> parsed = BackendUtil::from_string("cuda");
    REQUIRE(parsed.ok());
    REQUIRE(parsed.value() == Backend::Cuda);
    REQUIRE(BackendUtil::from_string("cpu").value() == Backend::Cpu);
    REQUIRE_FALSE(BackendUtil::from_string("gpu").ok());

#if defined(PARCAE_HAS_CUDA)
    REQUIRE(BackendUtil::cuda_built());
    REQUIRE(BackendUtil::ensure_usable(Backend::Cuda).ok());

    StatusOr<TransformEnvelope> env = TransformEnvelope::from_json(nlohmann::json{
        {"transform_id", "caesar"},
        {"direction", "decrypt"},
        {"params", {{"shift", 5}}},
    });
    REQUIRE(env.ok());

    const std::vector<Index29> cipher = {I(5), I(6), I(7), I(10)};
    StatusOr<std::vector<Index29>> cpu =
        ToolApi::apply_to_indices(cipher, env.value(), Backend::Cpu);
    StatusOr<std::vector<Index29>> cuda =
        ToolApi::apply_to_indices(cipher, env.value(), Backend::Cuda);
    REQUIRE(cpu.ok());
    REQUIRE(cuda.ok());
    REQUIRE(cuda.value() == cpu.value());

    StatusOr<double> cpu_score =
        ToolApi::score(test_ctx(), cpu.value(), "ic_mod29", "v0", {}, {}, Backend::Cpu);
    StatusOr<double> cuda_score =
        ToolApi::score(test_ctx(), cpu.value(), "ic_mod29", "v0", {}, {}, Backend::Cuda);
    REQUIRE(cpu_score.ok());
    REQUIRE(cuda_score.ok());
    REQUIRE(cuda_score.value() == cpu_score.value());
#else
    REQUIRE_FALSE(BackendUtil::cuda_built());
    REQUIRE_FALSE(BackendUtil::ensure_usable(Backend::Cuda).ok());
    StatusOr<TransformEnvelope> env = TransformEnvelope::from_json(nlohmann::json{
        {"transform_id", "identity"},
        {"direction", "decrypt"},
        {"params", nlohmann::json::object()},
    });
    REQUIRE(env.ok());
    REQUIRE_FALSE(
        ToolApi::apply_to_indices(std::vector<Index29>{I(1)}, env.value(), Backend::Cuda).ok());
#endif
}

TEST_CASE("tool::to_latin preferred labels", "[tool][latin]") {
    const auto ctx = test_ctx();
    // Index 0 preferred is typically F; 1 is U — join without spaces.
    StatusOr<std::string> latin = ToolApi::to_latin(ctx, std::vector<Index29>{I(0), I(1)});
    REQUIRE(latin.ok());
    REQUIRE_FALSE(latin.value().empty());
    REQUIRE(latin.value().find(' ') == std::string::npos);
}

TEST_CASE("tool::score ic and chi2", "[tool][score]") {
    const auto ctx = test_ctx();
    const std::vector<Index29> xs = {I(3), I(3), I(3), I(3)};
    StatusOr<double> ic = ToolApi::score(ctx, xs, "ic_mod29");
    REQUIRE(ic.ok());
    REQUIRE(ic.value() == Catch::Approx(1.0).margin(0.0));

    StatusOr<double> chi2 = ToolApi::score(ctx, xs, "chi2_english_gp_v0");
    REQUIRE(chi2.ok());
    REQUIRE(chi2.value() >= 0.0);
}

TEST_CASE("tool::validate_fixture by id and path", "[tool][validate]") {
    const auto ctx = test_ctx();

    const ValidationReport by_id =
        ToolApi::validate_fixture(ctx, "a-warning", /*require_locked=*/true);
    REQUIRE(by_id.fixture_id() == "a-warning");
    REQUIRE(by_id.ok());

    const auto path = (ctx.data_root() / "fixtures" / "solved" / "welcome").string();
    const ValidationReport by_path = ToolApi::validate_fixture(ctx, path, /*require_locked=*/true);
    REQUIRE(by_path.fixture_id() == "welcome");
    REQUIRE(by_path.ok());
}

TEST_CASE("tool::list registries", "[tool]") {
    const auto transforms = ToolApi::list_transform_ids();
    REQUIRE(transforms.size() >= 8);
    REQUIRE(std::find(transforms.begin(), transforms.end(), "caesar") != transforms.end());

    const auto scores = ToolApi::list_score_ids();
    REQUIRE(scores.size() == 6);
    REQUIRE(std::find(scores.begin(), scores.end(), "ic_mod29") != scores.end());
}

TEST_CASE("GenerateCandidates from_indices and from_source", "[tool][generate]") {
    const auto ctx = test_ctx();
    REQUIRE(GenerateCandidates::list_generator_ids().size() == 8);

    const std::vector<Index29> cipher = {I(0), I(5), I(10)};
    StatusOr<std::vector<TransformCandidate>> from_idx =
        GenerateCandidates::from_indices("gen_caesar", cipher);
    REQUIRE(from_idx.ok());
    REQUIRE(from_idx.value().size() == 29);
    REQUIRE(from_idx.value()[3].params().at("shift").get<int>() == 3);

    StatusOr<std::vector<TransformCandidate>> from_indices_text =
        GenerateCandidates::from_source(ctx, "gen_atbash", "0,5,10", "indices");
    REQUIRE(from_indices_text.ok());
    REQUIRE(from_indices_text.value().size() == 1);

    StatusOr<std::vector<TransformCandidate>> from_latin =
        GenerateCandidates::from_source(ctx, "gen_caesar", "ABC", "latin");
    REQUIRE(from_latin.ok());
    REQUIRE(from_latin.value().size() == 29);

    // Build a stream via indices path then wrap as TokenStream isn't needed —
    // from_stream is covered by applying registry on consumable indices of a
    // fixture ciphertext after tokenize.
    const auto cipher_path =
        ctx.data_root() / "fixtures" / "solved" / "a-warning" / "ciphertext.txt";
    std::ifstream in(cipher_path, std::ios::binary);
    REQUIRE(in);
    std::ostringstream buf;
    buf << in.rdbuf();
    StatusOr<TokenStream> stream = ToolApi::tokenize(ctx, buf.str());
    REQUIRE(stream.ok());
    StatusOr<std::vector<TransformCandidate>> from_stream =
        GenerateCandidates::from_stream("gen_atbash", stream.value());
    REQUIRE(from_stream.ok());
    REQUIRE(from_stream.value().size() == 1);

    REQUIRE_FALSE(GenerateCandidates::from_indices("gen_nope", cipher).ok());
    REQUIRE_FALSE(GenerateCandidates::from_source(ctx, "gen_caesar", "", "indices").ok());
}

TEST_CASE("RankCandidates top-k stable ties and JSON", "[tool][rank]") {
    const auto ctx = test_ctx();

    const std::vector<Index29> plain = {I(0), I(1), I(2), I(3)};
    StatusOr<std::vector<Index29>> cipher = [&]() {
        StatusOr<TransformEnvelope> env = TransformEnvelope::from_json(nlohmann::json{
            {"transform_id", "caesar"},
            {"direction", "encrypt"},
            {"params", {{"shift", 7}}},
        });
        REQUIRE(env.ok());
        return ToolApi::apply_to_indices(plain, env.value());
    }();
    REQUIRE(cipher.ok());

    StatusOr<std::vector<TransformCandidate>> candidates =
        GenerateCandidates::from_indices("gen_caesar", cipher.value());
    REQUIRE(candidates.ok());

    ScoreRequest request;
    request.reference = std::span<const Index29>(plain);

    StatusOr<BatchResult> ranked = RankCandidates::run(candidates.value(), "exact_match",
                                                       /*k=*/3, &ctx, request);
    REQUIRE(ranked.ok());
    REQUIRE(ranked.value().top().size() == 3);
    REQUIRE(ranked.value().top()[0].score() == 1.0);
    REQUIRE(ranked.value().top()[0].candidate_id() == "caesar:shift=7");

    // Equal non-matches: stable order by candidate_id then source_index.
    REQUIRE(ranked.value().top()[1].score() == 0.0);
    REQUIRE(ranked.value().top()[2].score() == 0.0);
    REQUIRE(ranked.value().top()[1].candidate_id() < ranked.value().top()[2].candidate_id());

    StatusOr<nlohmann::json> payload =
        RankCandidates::result_to_json(ranked.value(), candidates.value(), &ctx);
    REQUIRE(payload.ok());
    REQUIRE(payload.value().at("order").get<std::string>() == "desc");
    REQUIRE(payload.value().at("backend").get<std::string>() == "cpu");
    REQUIRE(payload.value().at("hits").size() == 3);
    REQUIRE(payload.value().at("hits").at(0).at("rank").get<std::size_t>() == 0);
    REQUIRE(payload.value().at("hits").at(0).at("envelope").is_object());
    REQUIRE(payload.value().at("hits").at(0).at("latin").is_string());

    REQUIRE_FALSE(RankCandidates::run(candidates.value(), "exact_match", 0, &ctx, request).ok());
    REQUIRE_FALSE(RankCandidates::run({}, "ic_mod29", 1, &ctx).ok());
    REQUIRE_FALSE(RankCandidates::run(candidates.value(), "chi2_english_gp_v0", 1, nullptr).ok());

    StatusOr<BatchResult> chi2 =
        RankCandidates::run(candidates.value(), "chi2_english_gp_v0", 2, &ctx);
    REQUIRE(chi2.ok());
    REQUIRE(chi2.value().top().size() == 2);
}

TEST_CASE("RankCandidates CUDA backend path", "[tool][rank][cuda]") {
    const auto ctx = test_ctx();

    const std::vector<Index29> plain = {I(0), I(1), I(2), I(3)};
    StatusOr<std::vector<Index29>> cipher =
        CaesarTransform{}.apply(plain, nlohmann::json{{"shift", 7}}, TransformDirection::Encrypt);
    REQUIRE(cipher.ok());

    StatusOr<std::vector<TransformCandidate>> candidates =
        GenerateCandidates::from_indices("gen_caesar", cipher.value());
    REQUIRE(candidates.ok());

    ScoreRequest request;
    request.reference = std::span<const Index29>(plain);

#if !defined(PARCAE_HAS_CUDA)
    StatusOr<BatchResult> no_cuda =
        RankCandidates::run(candidates.value(), "exact_match",
                            /*k=*/3, &ctx, request, nlohmann::json::object(), "v0",
                            BatchExecution::Serial, Backend::Cuda);
    REQUIRE_FALSE(no_cuda.ok());
    REQUIRE(no_cuda.status().message().find("CUDA") != std::string::npos);
#else
    if (!CudaScore::available()) {
        SKIP("No CUDA device");
    }

    StatusOr<BatchResult> cpu =
        RankCandidates::run(candidates.value(), "exact_match",
                            /*k=*/3, &ctx, request, nlohmann::json::object(), "v0",
                            BatchExecution::Serial, Backend::Cpu);
    REQUIRE(cpu.ok());

    StatusOr<BatchResult> cuda =
        RankCandidates::run(candidates.value(), "exact_match",
                            /*k=*/3, &ctx, request, nlohmann::json::object(), "v0",
                            BatchExecution::Serial, Backend::Cuda);
    REQUIRE(cuda.ok());
    REQUIRE(cuda.value().top().size() == cpu.value().top().size());
    for (std::size_t i = 0; i < cpu.value().top().size(); ++i) {
        REQUIRE(cuda.value().top()[i].candidate_id() == cpu.value().top()[i].candidate_id());
        REQUIRE(cuda.value().top()[i].score() == cpu.value().top()[i].score());
        REQUIRE(cuda.value().top()[i].source_index() == cpu.value().top()[i].source_index());
    }

    StatusOr<nlohmann::json> payload =
        RankCandidates::result_to_json(cuda.value(), candidates.value(), &ctx, 64, Backend::Cuda);
    REQUIRE(payload.ok());
    REQUIRE(payload.value().at("backend").get<std::string>() == "cuda");
#endif
}

TEST_CASE("generate+rank a-warning: atbash wins chi2 without plaintext reference",
          "[tool][generate][rank][a-warning]") {
    const auto ctx = test_ctx();
    const auto cipher_path =
        ctx.data_root() / "fixtures" / "solved" / "a-warning" / "ciphertext.txt";
    std::ifstream in(cipher_path, std::ios::binary);
    REQUIRE(in);
    std::ostringstream buf;
    buf << in.rdbuf();

    StatusOr<std::vector<TransformCandidate>> atbash =
        GenerateCandidates::from_source(ctx, "gen_atbash", buf.str(), "runes");
    REQUIRE(atbash.ok());
    REQUIRE(atbash.value().size() == 1);
    REQUIRE(atbash.value()[0].candidate_id() == "atbash");

    StatusOr<std::vector<TransformCandidate>> caesar =
        GenerateCandidates::from_source(ctx, "gen_caesar", buf.str(), "runes");
    REQUIRE(caesar.ok());
    REQUIRE(caesar.value().size() == 29);

    std::vector<TransformCandidate> pool;
    pool.reserve(1 + caesar.value().size());
    pool.push_back(atbash.value()[0]);
    for (const TransformCandidate& c : caesar.value()) {
        pool.push_back(c);
    }

    // Unary language score only — no ScoreRequest.reference / params.reference.
    StatusOr<BatchResult> ranked = RankCandidates::run(pool, "chi2_english_gp_v0", /*k=*/3, &ctx);
    REQUIRE(ranked.ok());
    REQUIRE(ranked.value().top().size() == 3);
    REQUIRE(ranked.value().top()[0].candidate_id() == "atbash");
    REQUIRE(ranked.value().top()[0].score() < ranked.value().top()[1].score());

    StatusOr<nlohmann::json> payload =
        RankCandidates::result_to_json(ranked.value(), pool, &ctx, /*latin_max_chars=*/32);
    REQUIRE(payload.ok());
    REQUIRE(payload.value().at("order").get<std::string>() == "asc");
    REQUIRE_FALSE(payload.value().contains("reference"));
    const std::string latin = payload.value().at("hits").at(0).at("latin").get<std::string>();
    REQUIRE(latin.rfind("AWARNING", 0) == 0);
}

class RankProgressRecordingSink : public ConsoleProgressSink {
public:
    void on_progress(const ConsoleProgressSnapshot& snapshot) override {
        std::lock_guard<std::mutex> lock(mutex_);
        ++progress_count;
        max_done = std::max(max_done, snapshot.candidates_done());
        last_rune_count = snapshot.rune_count();
        if (snapshot.best_score().has_value()) {
            last_best_label = snapshot.best_label();
        }
    }

    void on_stage(std::string_view stage, const ConsoleProgressSnapshot& snapshot) override {
        std::lock_guard<std::mutex> lock(mutex_);
        stages.emplace_back(stage);
        stage_total = snapshot.candidates_total();
    }

    std::mutex mutex_;
    std::size_t progress_count = 0;
    std::size_t max_done = 0;
    std::size_t last_rune_count = 0;
    std::string last_best_label;
    std::vector<std::string> stages;
    std::optional<std::size_t> stage_total;
};

TEST_CASE("RankCandidates forwards progress sink / top-k unchanged", "[tool][rank][progress]") {
    const auto ctx = test_ctx();

    const std::vector<Index29> plain = {I(0), I(1), I(2), I(3)};
    StatusOr<std::vector<Index29>> cipher =
        CaesarTransform{}.apply(plain, nlohmann::json{{"shift", 7}}, TransformDirection::Encrypt);
    REQUIRE(cipher.ok());

    StatusOr<std::vector<TransformCandidate>> candidates =
        GenerateCandidates::from_indices("gen_caesar", cipher.value());
    REQUIRE(candidates.ok());

    ScoreRequest request;
    request.reference = std::span<const Index29>(plain);

    RankProgressRecordingSink sink;
    BatchRunner::Progress progress;
    progress.sink = &sink;
    // rune_count left 0 → RankCandidates fills from candidate length (4).

    StatusOr<BatchResult> with_sink =
        RankCandidates::run(candidates.value(), "exact_match",
                            /*k=*/3, &ctx, request, nlohmann::json::object(), "v0",
                            BatchExecution::Serial, Backend::Cpu, progress);
    REQUIRE(with_sink.ok());

    StatusOr<BatchResult> without = RankCandidates::run(candidates.value(), "exact_match",
                                                        /*k=*/3, &ctx, request);
    REQUIRE(without.ok());

    REQUIRE(with_sink.value().top().size() == without.value().top().size());
    for (std::size_t i = 0; i < without.value().top().size(); ++i) {
        REQUIRE(with_sink.value().top()[i].candidate_id() ==
                without.value().top()[i].candidate_id());
        REQUIRE(with_sink.value().top()[i].score() == without.value().top()[i].score());
        REQUIRE(with_sink.value().top()[i].source_index() ==
                without.value().top()[i].source_index());
    }

    REQUIRE(sink.stages.size() == 1);
    REQUIRE(sink.stages[0] == "score");
    REQUIRE(sink.stage_total.has_value());
    REQUIRE(sink.stage_total.value() == 29);
    REQUIRE(sink.progress_count == 29);
    REQUIRE(sink.max_done == 29);
    REQUIRE(sink.last_rune_count == plain.size());
    REQUIRE(sink.last_best_label == "caesar:shift=7");
}
