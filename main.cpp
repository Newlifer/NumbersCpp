#include <iostream>
#include <immintrin.h>
#include <emmintrin.h>

using vecint_t = __m256i;

inline vecint_t negative(vecint_t value) {
        constexpr vecint_t __neg = {-1, -1, -1, -1};
        return _mm256_mul_epi32(value, __neg);
}

inline vecint_t numbers_lengths(vecint_t value) {
        return vecint_t{1, 1, 1, 1}; // FIXME implement it
}

struct i_binary_op {
        virtual vecint_t operator()(vecint_t left, vecint_t right) = 0;
};

struct add_op : i_binary_op {
        vecint_t operator()(vecint_t left, vecint_t right) final {
                return _mm256_add_epi32(left, right);
        }
};

struct sub_op : i_binary_op {
        vecint_t operator()(vecint_t left, vecint_t right) final {
                return _mm256_sub_epi32(left, right);
        }
};

struct mul_op : i_binary_op {
        vecint_t operator()(vecint_t left, vecint_t right) final {
                return _mm256_mul_epi32(left, right);
        }
};

struct concat_op : i_binary_op {
        vecint_t operator()(vecint_t left, vecint_t right) final {
                const auto shift_distances = numbers_lengths(right);
                constexpr vecint_t multiplier = {10, 10, 10, 10};
                const auto shifted = _mm256_mul_epi32(left, _mm256_mul_epi32(shift_distances, multiplier));
                return _mm256_add_epi32(shifted, right);
        }
};

int main() {
        constexpr vecint_t left = {127, 945, 767, 6868};
        constexpr vecint_t right = {5656, 4545, 67576, 987654321};
        sub_op substr;

        for(std::size_t i = 0; i < 100'000'000; ++i) {
                substr(left, right);
        }
        return 0;
}
