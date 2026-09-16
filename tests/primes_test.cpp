#include <parcae/math/primes.hpp>

#include <catch2/catch_test_macros.hpp>

#include <cstdint>
#include <vector>

TEST_CASE("Primes::sieve_upto returns primes <= limit", "[primes]") {
    REQUIRE(Primes::sieve_upto(0).empty());
    REQUIRE(Primes::sieve_upto(1).empty());
    REQUIRE(Primes::sieve_upto(2) == std::vector<std::uint64_t>{2});
    REQUIRE(Primes::sieve_upto(10) == std::vector<std::uint64_t>{2, 3, 5, 7});
    REQUIRE(
        Primes::sieve_upto(30) ==
        std::vector<std::uint64_t>{2, 3, 5, 7, 11, 13, 17, 19, 23, 29});
}

TEST_CASE("Primes::first and nth match the first N primes", "[primes]") {
    // First 25 primes (OEIS A000040); also covers Gematria Primus' first-29 prefix.
    const std::vector<std::uint64_t> first_25 = {
        2,  3,  5,  7,  11, 13, 17, 19, 23, 29, 31, 37, 41,
        43, 47, 53, 59, 61, 67, 71, 73, 79, 83, 89, 97};

    StatusOr<std::vector<std::uint64_t>> generated = Primes::first(first_25.size());
    REQUIRE(generated.ok());
    REQUIRE(generated.value() == first_25);

    for (std::size_t i = 0; i < first_25.size(); ++i) {
        StatusOr<std::uint64_t> p = Primes::nth(i);
        REQUIRE(p.ok());
        REQUIRE(p.value() == first_25[i]);
    }

    // Gematria Primus attaches the first 29 primes (2 … 109).
    StatusOr<std::vector<std::uint64_t>> first_29 = Primes::first(29);
    REQUIRE(first_29.ok());
    REQUIRE(first_29.value().front() == 2);
    REQUIRE(first_29.value().back() == 109);
    REQUIRE(first_29.value()[9] == 29);
    REQUIRE(first_29.value()[28] == 109);

    StatusOr<std::vector<std::uint64_t>> empty = Primes::first(0);
    REQUIRE(empty.ok());
    REQUIRE(empty.value().empty());
}

TEST_CASE("Primes::nth is deterministic for larger indices", "[primes]") {
    // Spot-checks beyond the gematria table (still small / exact).
    REQUIRE(Primes::nth(25).value() == 101);  // 26th prime
    REQUIRE(Primes::nth(99).value() == 541);  // 100th prime
    REQUIRE(Primes::nth(168).value() == 1009);
}
