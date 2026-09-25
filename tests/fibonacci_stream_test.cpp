#include <catch2/catch_test_macros.hpp>
#include <cstdint>
#include <parcae/core/index29.hpp>
#include <parcae/core/z29.hpp>
#include <parcae/math/fibonacci_stream.hpp>
#include <vector>

TEST_CASE("FibonacciStream classic first terms match known F_n mod 29",
          "[math][fibonacci]") {
    // Unreduced Fibonacci: 0,1,1,2,3,5,8,13,21,34,55,89,144,233,377,610,987,1597,2584,4181
    // Reduced mod 29:
    const std::vector<std::uint8_t> expected = {0, 1, 1, 2, 3, 5, 8, 13, 21, 5,
                                                26, 2, 28, 1, 0, 1, 1, 2, 3, 5};

    std::vector<Index29> generated = FibonacciStream::first(expected.size());
    REQUIRE(generated.size() == expected.size());
    for (std::size_t i = 0; i < expected.size(); ++i) {
        REQUIRE(generated[i].value() == expected[i]);
        REQUIRE(FibonacciStream::nth(i).value() == expected[i]);
    }

    REQUIRE(FibonacciStream::first(0).empty());
    REQUIRE(FibonacciStream::nth(0).value() == 0);
    REQUIRE(FibonacciStream::nth(1).value() == 1);
}

TEST_CASE("FibonacciStream satisfies recurrence mod 29", "[math][fibonacci]") {
    constexpr std::size_t n = 128;
    const std::vector<Index29> stream = FibonacciStream::first(n);
    for (std::size_t i = 2; i < n; ++i) {
        REQUIRE(stream[i] == Z29::add(stream[i - 2], stream[i - 1]));
        REQUIRE(FibonacciStream::nth(i) == stream[i]);
    }
}

TEST_CASE("FibonacciStream seeded starts from s0,s1", "[math][fibonacci]") {
    const Index29 s0{3};
    const Index29 s1{7};
    REQUIRE(FibonacciStream::nth_seeded(s0, s1, 0) == s0);
    REQUIRE(FibonacciStream::nth_seeded(s0, s1, 1) == s1);
    REQUIRE(FibonacciStream::nth_seeded(s0, s1, 2).value() == 10); // 3+7
    REQUIRE(FibonacciStream::nth_seeded(s0, s1, 3).value() == 17); // 7+10
    REQUIRE(FibonacciStream::nth_seeded(s0, s1, 4).value() == 27); // 10+17
    REQUIRE(FibonacciStream::nth_seeded(s0, s1, 5).value() == 15); // 17+27=44≡15

    const std::vector<Index29> first5 = FibonacciStream::first_seeded(s0, s1, 5);
    REQUIRE(first5.size() == 5);
    for (std::size_t i = 0; i < 5; ++i) {
        REQUIRE(first5[i] == FibonacciStream::nth_seeded(s0, s1, i));
    }

    // Classic seed (0,1) matches nth/first.
    REQUIRE(FibonacciStream::first_seeded(Index29{0}, Index29{1}, 10) ==
            FibonacciStream::first(10));
}

TEST_CASE("FibonacciStream classic spot-checks beyond the short prefix",
          "[math][fibonacci]") {
    // F_20 = 6765 → 6765 = 233·29 + 8
    REQUIRE(FibonacciStream::nth(20).value() == 8);
    // F_29 = 514229 → 514229 = 17732·29 + 1
    REQUIRE(FibonacciStream::nth(29).value() == 1);
    // Pisano period π(29) = 56: (F_56, F_57) ≡ (0, 1) (OEIS A001175).
    REQUIRE(FibonacciStream::nth(56).value() == 0);
    REQUIRE(FibonacciStream::nth(57).value() == 1);
    REQUIRE(FibonacciStream::nth(58).value() == 1); // F_2
}
