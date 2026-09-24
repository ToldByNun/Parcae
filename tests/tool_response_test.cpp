#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>
#include <parcae/tool/tool_response.hpp>
#include <sstream>
#include <string>

TEST_CASE("ToolErrorCodeUtil round-trips stable codes", "[tool][response]") {
    const ToolErrorCode codes[] = {
        ToolErrorCode::Usage,    ToolErrorCode::Io,       ToolErrorCode::Schema,
        ToolErrorCode::Policy,   ToolErrorCode::NotBuilt, ToolErrorCode::Validation,
        ToolErrorCode::Internal,
    };
    for (const ToolErrorCode code : codes) {
        StatusOr<ToolErrorCode> parsed =
            ToolErrorCodeUtil::from_string(ToolErrorCodeUtil::to_string(code));
        REQUIRE(parsed.ok());
        REQUIRE(parsed.value() == code);
    }
    REQUIRE_FALSE(ToolErrorCodeUtil::from_string("nope").ok());
}

TEST_CASE("ToolErrorCodeUtil exit_status maps validation soft vs hard", "[tool][response]") {
    REQUIRE(ToolErrorCodeUtil::exit_status(ToolErrorCode::Validation) == 1);
    REQUIRE(ToolErrorCodeUtil::exit_status(ToolErrorCode::Usage) == 2);
    REQUIRE(ToolErrorCodeUtil::exit_status(ToolErrorCode::Policy) == 2);
    REQUIRE(ToolErrorCodeUtil::exit_status(ToolErrorCode::NotBuilt) == 2);
    REQUIRE(ToolResponse::exit_failure(ToolErrorCode::Validation) == 1);
    REQUIRE(ToolResponse::exit_success() == 0);
}

TEST_CASE("ToolResponse::success builds valid envelope", "[tool][response]") {
    const nlohmann::json env = ToolResponse::success("score", std::string("cpu"),
                                                     nlohmann::json{
                                                         {"score_id", "chi2_english_gp_v0"},
                                                         {"score_version", "v0"},
                                                         {"value", 12.34},
                                                     });

    REQUIRE(ToolResponse::validate(env).ok());
    REQUIRE(env.at("schema").get<std::string>() == ToolResponse::schema_id);
    REQUIRE(env.at("ok").get<bool>() == true);
    REQUIRE(env.at("tool").get<std::string>() == "score");
    REQUIRE(env.at("backend").get<std::string>() == "cpu");
    REQUIRE(env.at("error").is_null());
    REQUIRE(env.at("result").at("value").get<double>() == Catch::Approx(12.34));
}

TEST_CASE("ToolResponse::success allows null backend", "[tool][response]") {
    const nlohmann::json env = ToolResponse::success(
        "catalog", std::nullopt, nlohmann::json{{"transforms", nlohmann::json::array()}});
    REQUIRE(ToolResponse::validate(env).ok());
    REQUIRE(env.at("backend").is_null());
}

TEST_CASE("ToolResponse::failure builds valid envelope", "[tool][response]") {
    const nlohmann::json env = ToolResponse::failure("decode", std::string("cpu"),
                                                     ToolErrorCode::Usage, "missing --input");

    REQUIRE(ToolResponse::validate(env).ok());
    REQUIRE(env.at("ok").get<bool>() == false);
    REQUIRE(env.at("result").is_null());
    REQUIRE(env.at("error").at("code").get<std::string>() == "usage");
    REQUIRE(env.at("error").at("message").get<std::string>() == "missing --input");
}

TEST_CASE("ToolResponse::write emits compact JSON line", "[tool][response]") {
    const nlohmann::json env =
        ToolResponse::success("tokenize", std::string("cpu"), nlohmann::json::object());
    std::ostringstream out;
    ToolResponse::write(out, env);
    const std::string text = out.str();
    REQUIRE_FALSE(text.empty());
    REQUIRE(text.back() == '\n');

    StatusOr<nlohmann::json> parsed = ToolResponse::parse(text);
    REQUIRE(parsed.ok());
    REQUIRE(parsed.value().at("tool").get<std::string>() == "tokenize");
}

TEST_CASE("ToolResponse::failure may carry optional details", "[tool][response]") {
    const nlohmann::json env =
        ToolResponse::failure("validate", std::nullopt, ToolErrorCode::Validation, "failed",
                              nlohmann::json{{"count", 1}});
    REQUIRE(ToolResponse::validate(env).ok());
    REQUIRE(env.at("error").at("details").at("count").get<int>() == 1);
}

TEST_CASE("ToolResponse::validate rejects mismatched ok/error/result", "[tool][response]") {
    nlohmann::json bad =
        ToolResponse::success("score", std::string("cpu"), nlohmann::json::object());
    bad["error"] = nlohmann::json{{"code", "usage"}, {"message", "x"}};
    REQUIRE_FALSE(ToolResponse::validate(bad).ok());

    bad = ToolResponse::failure("score", std::nullopt, ToolErrorCode::Io, "missing");
    bad["result"] = nlohmann::json::object();
    REQUIRE_FALSE(ToolResponse::validate(bad).ok());

    bad = ToolResponse::success("score", std::string("gpu"), nlohmann::json::object());
    REQUIRE_FALSE(ToolResponse::validate(bad).ok());

    REQUIRE_FALSE(ToolResponse::parse("{").ok());
}
