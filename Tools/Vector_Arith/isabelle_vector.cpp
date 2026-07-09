/* The Q1.15 SIMD kernel. Two processes load this same shared object: CPython
 * through ctypes.CDLL, and Isabelle/ML through Foreign.loadLibrary -- the latter
 * into a process with no interpreter in it.
 *
 * INVARIANT: nothing here may reference a CPython symbol, function or data.
 *   - Unix: Poly/ML dlopens with RTLD_LAZY (libpolyml/polyffi.cpp:167), so an
 *     undefined *function* would go unnoticed until called, but a data symbol
 *     relocates eagerly and breaks the load outright.
 *   - Windows: LoadLibrary (polyffi.cpp:153) resolves the whole ordinary import
 *     table at load. A python3.dll import may happen to resolve if Python is on the
 *     process PATH, but relying on that is a fragile, version-coupled coupling --
 *     the ML process is launched from a Cygwin shell whose PATH we do not control.
 *
 * The Python-facing glue lives in Isabelle_Semantic_Embedding/_vecgather.c, a
 * separate CPython extension module. Two guards keep this file honest:
 * -Wl,-z,defs makes a stray symbol a link error, and test_ml_dlopen reproduces
 * Poly/ML's load at build time.
 */
#include <stdint.h>
#include <stddef.h>
#include <stdlib.h>
#include <string.h>
// #include <queue>

#undef HWY_TARGET_INCLUDE
#define HWY_TARGET_INCLUDE "isabelle_vector-inl.h"
#include "hwy/foreach_target.h"
#include "hwy/highway.h"

#include HWY_TARGET_INCLUDE

namespace project {
HWY_EXPORT(DotQ15Impl);
}

extern "C" int16_t dot_q15(const int16_t* a, const int16_t* b, size_t N) {
  return HWY_DYNAMIC_DISPATCH(project::DotQ15Impl)(a, b, N);
}

struct Entry { int16_t score; int32_t index; };
struct Heap {
    int32_t size;
    int32_t capacity;
    Entry* data;
};

Heap allocate_heap(int32_t size) {
  Heap heap;
  heap.size = 0;
  heap.capacity = size;
  Entry* raw = (Entry*)malloc((size + 1) * sizeof(Entry));
  heap.data = raw - 1; /* 1-indexed: data[1..size+1] maps to raw[0..size] */
  return heap;
}

void free_heap(Heap& heap) {
  free(heap.data + 1); /* restores original malloc pointer */
}

/* Min-heap: root = smallest score = gatekeeper for top-k */
static void sift_up(Heap& heap, int32_t i) {
  while (i > 1 && heap.data[i].score < heap.data[i / 2].score) {
    std::swap(heap.data[i], heap.data[i / 2]);
    i /= 2;
  }
}

static void sift_down(Heap& heap) {
  int32_t i = 1;
  while (true) {
    int32_t left = 2 * i;
    int32_t right = 2 * i + 1;
    int32_t smallest = i;
    if (left <= heap.size && heap.data[left].score < heap.data[smallest].score)
      smallest = left;
    if (right <= heap.size && heap.data[right].score < heap.data[smallest].score)
      smallest = right;
    if (smallest == i)
      break;
    std::swap(heap.data[i], heap.data[smallest]);
    i = smallest;
  }
}

Entry pop_heap(Heap& heap) {
  Entry result = heap.data[1];
  heap.data[1] = heap.data[heap.size];
  heap.size--;
  sift_down(heap);
  return result;
}

/* Offer one (score,index) to the top-k min-heap. Shared by both entry points. */
static inline void heap_offer(Heap& heap, int16_t score, int32_t index) {
  if (heap.size < heap.capacity) {
    /* Heap not full: insert at end, sift up */
    heap.size++;
    heap.data[heap.size] = {score, index};
    sift_up(heap, heap.size);
  } else if (score > heap.data[1].score) {
    /* New score beats the weakest in top-k: replace root, sift down */
    heap.data[1] = {score, index};
    sift_down(heap);
  }
  /* else: worse than weakest, skip */
}

/* Drain the min-heap into result arrays in DESCENDING score order.
   Pop yields ascending; fill in reverse. Consumes exactly heap.size==k entries. */
static void heap_fill_desc(Heap& heap, int32_t k, int32_t* out_idx, int16_t* out_scores) {
  for (int32_t i = k - 1; i >= 0; i--) {
    Entry entry = pop_heap(heap);
    out_idx[i] = entry.index;
    out_scores[i] = entry.score;
  }
}

extern "C" void top_k_q15(const int16_t* vectors, const int16_t* query, int32_t D, int32_t N, int32_t k, int32_t* result_indexes, int16_t* result_scores) {
  int16_t* scores = (int16_t*)malloc(N * sizeof(int16_t));
  int32_t D_aligned = (D + 31) & ~31;
  for (int32_t i = 0; i < N; i++)
    scores[i] = dot_q15(vectors + i * D_aligned, query, D);
  Heap heap = allocate_heap(k);
  for (int32_t i = 0; i < N; i++)
    heap_offer(heap, scores[i], i);
  heap_fill_desc(heap, k, result_indexes, result_scores);
  free(scores);
  free_heap(heap);
}

/* Gather-mode top-k, ZERO-COPY. vec_addrs[i] points directly at the i-th Q1.15
   vector (D int16, packed, arbitrary alignment) — typically straight into an LMDB
   mmap value. The kernel reads it in place via LoadU; nothing is copied.

   Measured: zero-copy LoadU beats memcpy-into-an-aligned-row by 1.5x (2.0x when
   the addresses are in scattered/unsorted order), because memcpy makes the data
   cross memory twice. Alignment itself is worth nothing here (aligned Load 75.5ms
   vs unaligned LoadU 76.0ms), so we neither pad the stored values nor copy.

   CALLER CONTRACT:
     - each vec_addrs[i] must have exactly D*2 readable bytes (assert len==D*2 on
       the Python side; a stale float32 record is D*4 and would silently mis-read);
     - the LMDB read transaction must stay open for the whole call, since the
       addresses point into its MVCC snapshot of the mmap.
   Returns 0 on success, 2 on invalid k/N. */
extern "C" int top_k_q15_gather(
    const uintptr_t* vec_addrs, const int16_t* query,
    int32_t D, int32_t N, int32_t k,
    int32_t* out_idx, int16_t* out_scores) {
  if (k > N) k = N;              /* F: clamp at entry, do not trust caller */
  if (k <= 0 || N <= 0) return 2;
  Heap heap = allocate_heap(k);
  for (int32_t i = 0; i < N; i++) {
    int16_t s = HWY_DYNAMIC_DISPATCH(project::DotQ15Impl)(
        (const int16_t*)vec_addrs[i], query, (size_t)D);   /* zero-copy, LoadU */
    heap_offer(heap, s, i);
  }
  heap_fill_desc(heap, k, out_idx, out_scores);
  free_heap(heap);
  return 0;
}