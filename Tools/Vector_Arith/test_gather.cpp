/* Tests for the SIMD dot + zero-copy gather path.
   test_topk.cpp only covers the scalar heap logic; this file is the first coverage
   of dot_q15 / top_k_q15_gather actually running Highway SIMD over real memory,
   including deliberately unaligned sources (the production case: LMDB mmap values
   sit at page_base+16, never 32/64B-aligned). */
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <math.h>

extern "C" int16_t dot_q15(const int16_t* a, const int16_t* b, size_t N);
extern "C" int top_k_q15_gather(const uintptr_t* vec_addrs, const int16_t* query,
                                int32_t D, int32_t N, int32_t k,
                                int32_t* out_idx, int16_t* out_scores);

static const double NORM = 0.95;

/* A self-contained LCG (Numerical Recipes' constants). rand_r is POSIX-only, so
   MinGW has none; and its sequence varies by libc, which would make these vectors
   -- and any failure they provoke -- differ between platforms. */
static unsigned next_rand(unsigned* seed) {
  *seed = *seed * 1664525u + 1013904223u;
  return *seed >> 1; /* drop the low bit: an LCG's lowest bits have short periods */
}

/* Quantize a random unit-ish vector to Q1.15 scaled to NORM, as production does. */
static void make_vec(int16_t* dst, int32_t D, unsigned* seed) {
  double* f = (double*)malloc(D * sizeof(double));
  double ss = 0.0;
  for (int32_t i = 0; i < D; i++) {
    double u = (double)next_rand(seed) / (double)0x7FFFFFFFu * 2.0 - 1.0;
    f[i] = u; ss += u * u;
  }
  double n = sqrt(ss);
  for (int32_t i = 0; i < D; i++)
    dst[i] = (int16_t)lrint(NORM * f[i] / n * 32768.0);
  free(f);
}

/* Reference: exactly the kernel's semantics — MulFixedPoint15 rounds, i.e.
   (a*b + 0x4000) >> 15 (this is what VPMULHRSW computes). Accumulate wide; the
   int16 accumulator cannot overflow for NORM-normalized inputs, since
   |s| <= NORM^2 * 32768 ~= 29573 by Cauchy-Schwarz. */
static int32_t ref_dot(const int16_t* a, const int16_t* b, int32_t D) {
  int32_t s = 0;
  for (int32_t i = 0; i < D; i++)
    s += (int32_t)((((int32_t)a[i] * (int32_t)b[i]) + 16384) >> 15);
  return s;
}

static int failures = 0;
static void check(int cond, const char* what) {
  printf("  %-52s %s\n", what, cond ? "PASS" : "FAIL");
  if (!cond) failures++;
}

int main(void) {
  unsigned seed = 12345;
  const int32_t D = 256, N = 500, K = 10;

  /* Deliberately UNALIGNED backing store: offset the base by one int16 so every
     row lands at addr % 64 != 0, mirroring LMDB's page_base+16 values. */
  int16_t* raw = (int16_t*)malloc((size_t)(N * D + 1) * sizeof(int16_t));
  int16_t* dom = raw + 1;
  for (int32_t i = 0; i < N; i++) make_vec(dom + (size_t)i * D, D, &seed);

  int16_t* qraw = (int16_t*)malloc((size_t)(D + 1) * sizeof(int16_t));
  int16_t* q = qraw + 1;                       /* unaligned query too */
  make_vec(q, D, &seed);

  printf("SIMD dot + zero-copy gather tests (D=%d, N=%d, k=%d)\n", D, N, K);
  printf("  domain base %% 64 = %zu, query %% 64 = %zu (both intentionally unaligned)\n",
         (size_t)((uintptr_t)dom % 64), (size_t)((uintptr_t)q % 64));

  /* 1) dot_q15 on unaligned inputs must match the floor reference. */
  int mism = 0;
  for (int32_t i = 0; i < N; i++) {
    int32_t r = ref_dot(dom + (size_t)i * D, q, D);
    if ((int16_t)r != dot_q15(dom + (size_t)i * D, q, D)) mism++;
  }
  check(mism == 0, "dot_q15 unaligned == rounding reference");

  /* 2) zero-copy gather: addresses point straight at each (unaligned) row. */
  uintptr_t* addrs = (uintptr_t*)malloc((size_t)N * sizeof(uintptr_t));
  for (int32_t i = 0; i < N; i++) addrs[i] = (uintptr_t)(dom + (size_t)i * D);
  int32_t* oi = (int32_t*)malloc((size_t)K * sizeof(int32_t));
  int16_t* os = (int16_t*)malloc((size_t)K * sizeof(int16_t));
  check(top_k_q15_gather(addrs, q, D, N, K, oi, os) == 0, "top_k_q15_gather returns 0");

  /* returned score must equal the reference dot of the returned index */
  int bad = 0;
  for (int32_t r = 0; r < K; r++)
    if (os[r] != (int16_t)ref_dot(dom + (size_t)oi[r] * D, q, D)) bad++;
  check(bad == 0, "returned scores match their indices' dots");

  /* scores must be descending, and no candidate may beat the k-th */
  int desc = 1;
  for (int32_t r = 1; r < K; r++) if (os[r] > os[r - 1]) desc = 0;
  check(desc, "scores are in descending order");

  int better = 0;
  for (int32_t i = 0; i < N; i++) {
    int in_top = 0;
    for (int32_t r = 0; r < K; r++) if (oi[r] == i) in_top = 1;
    if (!in_top && (int16_t)ref_dot(dom + (size_t)i * D, q, D) > os[K - 1]) better++;
  }
  check(better == 0, "no excluded candidate beats the k-th score");

  /* 3) entry guards */
  check(top_k_q15_gather(addrs, q, D, 5, 999, oi, os) == 0, "k > N is clamped, not OOB");
  check(top_k_q15_gather(addrs, q, D, 0, 5, oi, os) == 2, "N == 0 rejected with rc=2");
  check(top_k_q15_gather(addrs, q, D, N, 0, oi, os) == 2, "k == 0 rejected with rc=2");

  free(raw); free(qraw); free(addrs); free(oi); free(os);
  printf("\n%s\n", failures == 0 ? "ALL PASS" : "FAILURES PRESENT");
  return failures == 0 ? 0 : 1;
}
