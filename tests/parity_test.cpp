#include <parcae/core/index29.hpp>
#include <parcae/interrupt/policy.hpp>
#include <parcae/parity/parity_record.hpp>
#include <parcae/transform/transform_direction.hpp>
#include <parcae/transform/transform_id.hpp>

#include <catch2/catch_test_macros.hpp>

#include <nlohmann/json.hpp>

#include <vector>

TEST_CASE("ParityRecord capture hashes params input output interrupt", "[parity]") {
    const std::vector<Index29> input{Index29{0}, Index29{1}, Index29{2}};
    const std::vector<Index29> output{Index29{3}, Index29{4}, Index29{5}};
    const nlohmann::json params{{"shift", 3}};
    StatusOr<InterruptPolicy> interrupt = InterruptPolicy::from_skip_indices({1});
    REQUIRE(interrupt.ok());

    const ParityRecord record = ParityRecord::capture(
        "caesar", params, interrupt.value(), input, output, "cpu");

    REQUIRE(record.transform_id() == "caesar");
    REQUIRE(record.backend() == "cpu");
    REQUIRE(record.params_hash_sha256() == ParityRecord::hash_json(params));
    REQUIRE(record.input_sha256() == ParityRecord::hash_indices(input));
    REQUIRE(record.output_sha256() == ParityRecord::hash_indices(output));
    REQUIRE(
        record.interrupt_sha256() == ParityRecord::hash_json(interrupt.value().to_json()));
    REQUIRE(record.params_hash_sha256().size() == 64);
    REQUIRE(record.output_sha256().size() == 64);
}

TEST_CASE("ParityRecord JSON round-trip", "[parity]") {
    const std::vector<Index29> xs{Index29{7}, Index29{8}};
    const ParityRecord original = ParityRecord::capture(
        "identity",
        nlohmann::json::object(),
        InterruptPolicy::none(),
        xs,
        xs);

    const nlohmann::json json = original.to_json();
    REQUIRE(json.at("parity_schema").get<std::string>() == ParityRecord::schema_id);

    StatusOr<ParityRecord> parsed = ParityRecord::from_json(json);
    REQUIRE(parsed.ok());
    REQUIRE(parsed.value() == original);
}

TEST_CASE("ParityRecord apply_and_capture matches ApplyTransform", "[parity]") {
    const std::vector<Index29> plain{Index29{1}, Index29{2}, Index29{3}, Index29{4}};
    const nlohmann::json params{{"shift", 5}};

    StatusOr<std::pair<std::vector<Index29>, ParityRecord>> ran =
        ParityRecord::apply_and_capture(
            TransformId::caesar(),
            plain,
            params,
            TransformDirection::Encrypt);
    REQUIRE(ran.ok());

    StatusOr<std::vector<Index29>> direct = ApplyTransform::apply(
        TransformId::caesar(), plain, params, TransformDirection::Encrypt);
    REQUIRE(direct.ok());
    REQUIRE(ran.value().first == direct.value());
    REQUIRE(ran.value().second.output_sha256() == ParityRecord::hash_indices(direct.value()));
    REQUIRE(ran.value().second.transform_id() == "caesar");
}

TEST_CASE("ParityRecord distinguishes different outputs", "[parity]") {
    const std::vector<Index29> input{Index29{0}, Index29{1}};
    const auto a = ParityRecord::capture(
        "atbash", nlohmann::json::object(), InterruptPolicy::none(), input, input);
    const std::vector<Index29> flipped{Index29{28}, Index29{27}};
    const auto b = ParityRecord::capture(
        "atbash", nlohmann::json::object(), InterruptPolicy::none(), input, flipped);
    REQUIRE(a.input_sha256() == b.input_sha256());
    REQUIRE(a.output_sha256() != b.output_sha256());
}
