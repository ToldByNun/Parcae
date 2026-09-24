#include "parcae/core/index29.hpp"
#include "parcae/corpus/fixture.hpp"
#include "parcae/corpus/fixture_loader.hpp"
#include "parcae/corpus/separator_grammar.hpp"
#include "parcae/corpus/tokenizer.hpp"
#include "parcae/gematria/gematria_profile_loader.hpp"
#include "parcae/interrupt/policy.hpp"
#include "parcae/parity/parity_record.hpp"
#include "parcae/transform/transform_direction.hpp"
#include "parcae/transform/transform_id.hpp"

#include "backend.hpp"

#include <catch2/catch_test_macros.hpp>
#include <string>
#include <utility>
#include <vector>

#ifndef PARCAE_TEST_DATA_DIR
#error "PARCAE_TEST_DATA_DIR must be defined"
#endif

/// Host helpers for locked solved-fixture CPU↔CUDA consumable parity (no C++ namespaces).
class SolvedCudaParity {
public:
    [[nodiscard]] static std::string fixture_dir(const char* id) {
        return std::string(PARCAE_TEST_DATA_DIR) + "/fixtures/solved/" + id;
    }

    [[nodiscard]] static GematriaProfile load_profile() {
        StatusOr<GematriaProfile> profile = GematriaProfileLoader::load_from_file(
            std::string(PARCAE_TEST_DATA_DIR) + "/profiles/gematria/gematria-primus-v0.json");
        REQUIRE(profile.ok());
        return profile.value();
    }

    [[nodiscard]] static SeparatorGrammar load_grammar() {
        StatusOr<SeparatorGrammar> grammar =
            SeparatorGrammar::load_from_file(std::string(PARCAE_TEST_DATA_DIR) +
                                             "/profiles/separators/rtkd-separator-grammar-v0.json");
        REQUIRE(grammar.ok());
        return grammar.value();
    }

    [[nodiscard]] static bool digests_match_except_backend(const ParityRecord& cpu,
                                                           const ParityRecord& cuda) {
        return cpu.transform_id() == cuda.transform_id() &&
               cpu.params_hash_sha256() == cuda.params_hash_sha256() &&
               cpu.input_sha256() == cuda.input_sha256() &&
               cpu.output_sha256() == cuda.output_sha256() &&
               cpu.interrupt_sha256() == cuda.interrupt_sha256() && cpu.backend() == "cpu" &&
               cuda.backend() == "cuda";
    }

    struct Case {
        Fixture fixture;
        TransformId id;
        TransformDirection direction{};
        InterruptPolicy interrupt;
        std::vector<Index29> consumable;
    };

    [[nodiscard]] static Case load_case(const char* id, const GematriaProfile& profile,
                                        const SeparatorGrammar& grammar) {
        StatusOr<Fixture> fixture = FixtureLoader::load_directory(fixture_dir(id));
        REQUIRE(fixture.ok());
        REQUIRE(fixture.value().verification_status() == "locked");

        StatusOr<TokenStream> stream =
            Tokenizer{profile, grammar}.tokenize(fixture.value().ciphertext(), true);
        REQUIRE(stream.ok());
        REQUIRE(stream.value().consumable_count() > 0);

        StatusOr<InterruptPolicy> interrupt =
            InterruptPolicy::from_skip_indices(fixture.value().skip_indices());
        REQUIRE(interrupt.ok());
        for (std::size_t skip : fixture.value().skip_indices()) {
            REQUIRE(skip < stream.value().consumable_count());
        }

        StatusOr<TransformId> transform_id =
            TransformId::from_string(fixture.value().transform_id());
        REQUIRE(transform_id.ok());
        StatusOr<TransformDirection> direction =
            TransformDirectionUtil::from_string(fixture.value().direction());
        REQUIRE(direction.ok());

        return Case{
            fixture.value(),
            transform_id.value(),
            direction.value(),
            interrupt.value(),
            stream.value().consumable_indices(),
        };
    }

private:
    SolvedCudaParity() = delete;
};

TEST_CASE("Solved-fixture consumable streams: CPU apply locks digests", "[cuda][parity][solved]") {
    const GematriaProfile profile = SolvedCudaParity::load_profile();
    const SeparatorGrammar grammar = SolvedCudaParity::load_grammar();

    for (const char* id : {
             "a-warning",
             "some-wisdom",
             "loss-of-divinity",
             "an-instruction",
             "koan-1",
             "welcome",
             "koan-2",
             "an-end",
             "lp2-57-identity",
         }) {
        SECTION(id) {
            const SolvedCudaParity::Case loaded = SolvedCudaParity::load_case(id, profile, grammar);

            StatusOr<std::pair<std::vector<Index29>, ParityRecord>> cpu =
                ParityRecord::apply_and_capture(loaded.id, loaded.consumable,
                                                loaded.fixture.params(), loaded.direction,
                                                loaded.interrupt, "cpu");
            REQUIRE(cpu.ok());
            REQUIRE(cpu.value().second.backend() == "cpu");
            REQUIRE(cpu.value().first.size() == loaded.consumable.size());
            REQUIRE(cpu.value().second.input_sha256() ==
                    ParityRecord::hash_indices(loaded.consumable));
            REQUIRE(cpu.value().second.output_sha256() ==
                    ParityRecord::hash_indices(cpu.value().first));
        }
    }
}

TEST_CASE("Solved-fixture consumable streams: CUDA matches CPU output_sha256",
          "[cuda][parity][solved]") {
#if !defined(PARCAE_HAS_CUDA)
    SUCCEED("Build with PARCAE_BUILD_CUDA=ON to exercise solved consumable CUDA parity");
    return;
#else
    if (!CudaBackend::available()) {
        SUCCEED("CUDA build present but no usable device; skipping solved CUDA apply");
        return;
    }

    const GematriaProfile profile = SolvedCudaParity::load_profile();
    const SeparatorGrammar grammar = SolvedCudaParity::load_grammar();

    for (const char* id : {
             "a-warning",
             "some-wisdom",
             "loss-of-divinity",
             "an-instruction",
             "koan-1",
             "welcome",
             "koan-2",
             "an-end",
             "lp2-57-identity",
         }) {
        SECTION(id) {
            const SolvedCudaParity::Case loaded = SolvedCudaParity::load_case(id, profile, grammar);

            StatusOr<std::pair<std::vector<Index29>, ParityRecord>> cpu =
                ParityRecord::apply_and_capture(loaded.id, loaded.consumable,
                                                loaded.fixture.params(), loaded.direction,
                                                loaded.interrupt, "cpu");
            REQUIRE(cpu.ok());

            StatusOr<std::pair<std::vector<Index29>, ParityRecord>> cuda =
                CudaBackend::apply_and_capture(loaded.id, loaded.consumable,
                                               loaded.fixture.params(), loaded.direction,
                                               loaded.interrupt);
            REQUIRE(cuda.ok());
            REQUIRE(cuda.value().first == cpu.value().first);
            REQUIRE(SolvedCudaParity::digests_match_except_backend(cpu.value().second,
                                                                   cuda.value().second));
            REQUIRE(cuda.value().second.output_sha256() == cpu.value().second.output_sha256());
        }
    }
#endif
}
