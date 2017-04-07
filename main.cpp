#include <iostream>
#include <immintrin.h>
#include <emmintrin.h>

using vecint_t = __m256i;

inline vecint_t negative(vecint_t value) {
        constexpr vecint_t __neg = {-1, -1, -1, -1};
        return _mm256_mul_epi32(value, __neg);
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
                return _mm256_add_epi32(left, negative(right));
        }
};

struct mul_op : i_binary_op {
        vecint_t operator()(vecint_t left, vecint_t right) final {
                return _mm256_mul_epi32(left, right);
        }
};

struct div_op : i_binary_op {
        vecint_t operator()(vecint_t left, vecint_t right) final {
                return _mm256_mul_epi32(left, negative(right));
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