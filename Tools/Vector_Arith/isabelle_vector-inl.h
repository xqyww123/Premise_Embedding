// isabelle_vector-inl.h - Highway SIMD implementation for Q15 dot product
#include <stdint.h>
#include <stddef.h>
#include "hwy/highway.h"

HWY_BEFORE_NAMESPACE();
namespace project {
namespace HWY_NAMESPACE {

namespace hwy = ::hwy::HWY_NAMESPACE;

int16_t DotQ15Impl(const int16_t* a, const int16_t* b, size_t N) {
  const hwy::ScalableTag<int16_t> d16;

  const size_t L = hwy::Lanes(d16);
  auto vacc = hwy::Zero(d16);  // int16 累加器

  size_t i = 0;
  for (; i + L <= N; i += L) {
    auto va16 = hwy::Load(d16, a + i);
    auto vb16 = hwy::Load(d16, b + i);
    auto prod16 = hwy::MulHigh(va16, vb16);
    vacc = hwy::Add(vacc, prod16);
  }

  // 剩余部分用 MaskedLoad
  const size_t remaining = N - i;
  if (remaining) {
    auto m = hwy::FirstN(d16, remaining);
    auto va16 = hwy::MaskedLoad(m, d16, a + i);
    auto vb16 = hwy::MaskedLoad(m, d16, b + i);
    auto prod16 = hwy::MulHigh(va16, vb16);
    vacc = hwy::Add(vacc, prod16);
  }
  return hwy::ReduceSum(d16, vacc);
}

}  // namespace HWY_NAMESPACE
}  // namespace project
HWY_AFTER_NAMESPACE();

