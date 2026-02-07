#include <stdint.h>
#include <stddef.h>
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
  heap.data = (Entry*)malloc((size + 1) * sizeof(Entry)) - 1;
  return heap;
}

void free_heap(Heap& heap) {
  free(heap.data);
}

void push_heap(Heap& heap, int16_t score, int32_t index) {
  int32_t i = heap.size + 1;
  heap.data[i] = {score, index};
  while (i > 1 && heap.data[i].score < heap.data[i / 2].score) {
    std::swap(heap.data[i], heap.data[i / 2]);
    i /= 2;
  }
  if (heap.size < heap.capacity)
    heap.size++;
}

Entry pop_heap(Heap& heap) {
  Entry result = heap.data[1];
  heap.data[1] = heap.data[heap.size];
  heap.size--;
  int32_t i = 1;
  while (i < heap.size) {
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
  return result;
}

extern "C" void top_k_q15(const int16_t* vectors, const int16_t* query, int32_t D, int32_t N, int32_t k, int32_t* result_indexes, int16_t* result_scores) {
  int16_t* scores = (int16_t*)malloc(N * sizeof(int16_t));
  int32_t D_aligned = (D + 31) & 31
  for (int32_t i = 0; i < N; i++)
    scores[i] = dot_q15(vectors + i * D_aligned, query, D);
  Heap heap = allocate_heap(k);
  for (int32_t i = 0; i < N; i++)
    push_heap(heap, -scores[i], i);
  for (int32_t i = 0; i < k; i++) {
    Entry entry = pop_heap(heap); 
    result_indexes[i] = entry.index;
    result_scores[i] = -entry.score;
  }
  free(scores);
  free_heap(heap);
}