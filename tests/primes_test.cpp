#include <catch2/catch_test_macros.hpp>
#include <cstdint>
#include <parcae/math/primes.hpp>
#include <parcae/math/totient_keystream.hpp>
#include <vector>

TEST_CASE("Primes::sieve_upto returns primes <= limit", "[primes]") {
    REQUIRE(Primes::sieve_upto(0).empty());
    REQUIRE(Primes::sieve_upto(1).empty());
    REQUIRE(Primes::sieve_upto(2) == std::vector<std::uint64_t>{2});
    REQUIRE(Primes::sieve_upto(10) == std::vector<std::uint64_t>{2, 3, 5, 7});
    REQUIRE(Primes::sieve_upto(30) ==
            std::vector<std::uint64_t>{2, 3, 5, 7, 11, 13, 17, 19, 23, 29});
}

TEST_CASE("Primes::first and nth match the first N primes", "[primes]") {
    // First 25 primes (OEIS A000040); also covers Gematria Primus' first-29 prefix.
    const std::vector<std::uint64_t> first_25 = {2,  3,  5,  7,  11, 13, 17, 19, 23, 29, 31, 37, 41,
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
    REQUIRE(Primes::nth(25).value() == 101); // 26th prime
    REQUIRE(Primes::nth(99).value() == 541); // 100th prime
    REQUIRE(Primes::nth(168).value() == 1009);
}

TEST_CASE("Primes cache survives growth and repeated nth", "[primes][cache]") {
    // Grow past small tables, then read earlier indices (cache must not corrupt).
    REQUIRE(Primes::nth(168).value() == 1009);
    REQUIRE(Primes::nth(0).value() == 2);
    REQUIRE(Primes::nth(25).value() == 101);

    StatusOr<std::vector<std::uint64_t>> first_29 = Primes::first(29);
    REQUIRE(first_29.ok());
    REQUIRE(first_29.value().front() == 2);
    REQUIRE(first_29.value().back() == 109);

    // Idempotent reads after cache fill.
    REQUIRE(Primes::nth(99).value() == 541);
    REQUIRE(Primes::nth(99).value() == 541);

    StatusOr<std::vector<std::uint64_t>> empty = Primes::first(0);
    REQUIRE(empty.ok());
    REQUIRE(empty.value().empty());
    REQUIRE(Primes::nth(0).value() == 2);
}

TEST_CASE("TotientKeystream first shifts are 1,2,4,6,10,... and wrap", "[primes][totient]") {
    // p: 2,3,5,7,11 → (p-1)%29 = 1,2,4,6,10
    StatusOr<std::vector<Index29>> prefix = TotientKeystream::shifts(5);
    REQUIRE(prefix.ok());
    REQUIRE(prefix.value() ==
            std::vector<Index29>{Index29{1}, Index29{2}, Index29{4}, Index29{6}, Index29{10}});

    // Through first wrap: …23→22, 29→28, 31→1
    StatusOr<std::vector<Index29>> through_wrap = TotientKeystream::shifts(11);
    REQUIRE(through_wrap.ok());
    REQUIRE(through_wrap.value() == std::vector<Index29>{Index29{1}, Index29{2}, Index29{4},
                                                         Index29{6}, Index29{10}, Index29{12},
                                                         Index29{16}, Index29{18}, Index29{22},
                                                         Index29{28}, Index29{1}});

    REQUIRE(TotientKeystream::from_prime(29).value() == 28);
    REQUIRE(TotientKeystream::from_prime(31).value() == 1);
    REQUIRE(TotientKeystream::from_prime(37).value() == 7);

    StatusOr<Index29> at_ten = TotientKeystream::shift_at(10); // p10=31
    REQUIRE(at_ten.ok());
    REQUIRE(at_ten.value() == Index29{1});

    StatusOr<std::vector<Index29>> offset = TotientKeystream::shifts(2, /*prime_start_index=*/9);
    REQUIRE(offset.ok());
    REQUIRE(offset.value() == std::vector<Index29>{Index29{28}, Index29{1}});
}
