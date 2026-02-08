#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <assert.h>

/* ---- 从 isabelle_vector.cpp 复制核心逻辑 ---- */

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
  free(heap.data + 1);
}

static void sift_up(Heap& heap, int32_t i) {
  while (i > 1 && heap.data[i].score < heap.data[i / 2].score) {
    Entry tmp = heap.data[i];
    heap.data[i] = heap.data[i / 2];
    heap.data[i / 2] = tmp;
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
    Entry tmp = heap.data[i];
    heap.data[i] = heap.data[smallest];
    heap.data[smallest] = tmp;
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

/* top_k 的纯逻辑版本（不调用 dot_q15，直接用预计算的 scores） */
void top_k_from_scores(int16_t* scores, int32_t N, int32_t k,
                       int32_t* result_indexes, int16_t* result_scores) {
  Heap heap = allocate_heap(k);
  for (int32_t i = 0; i < N; i++) {
    if (heap.size < heap.capacity) {
      heap.size++;
      heap.data[heap.size] = {scores[i], i};
      sift_up(heap, heap.size);
    } else if (scores[i] > heap.data[1].score) {
      heap.data[1] = {scores[i], i};
      sift_down(heap);
    }
  }
  for (int32_t i = k - 1; i >= 0; i--) {
    Entry entry = pop_heap(heap);
    result_indexes[i] = entry.index;
    result_scores[i] = entry.score;
  }
  free_heap(heap);
}

/* ---- 测试 ---- */

/* 对 scores 排序取 top-k 作为 ground truth */
void brute_force_top_k(int16_t* scores, int32_t N, int32_t k,
                       int32_t* expected_indexes, int16_t* expected_scores) {
  /* 构建 (score, index) 对并排序 */
  int32_t* order = (int32_t*)malloc(N * sizeof(int32_t));
  for (int32_t i = 0; i < N; i++) order[i] = i;
  /* 简单选择排序（测试用，不需要高效） */
  for (int32_t i = 0; i < k && i < N; i++) {
    int32_t best = i;
    for (int32_t j = i + 1; j < N; j++) {
      if (scores[order[j]] > scores[order[best]])
        best = j;
    }
    int32_t tmp = order[i]; order[i] = order[best]; order[best] = tmp;
  }
  for (int32_t i = 0; i < k; i++) {
    expected_indexes[i] = order[i];
    expected_scores[i] = scores[order[i]];
  }
  free(order);
}

int test_case(const char* name, int16_t* scores, int32_t N, int32_t k) {
  int32_t* expect_idx = (int32_t*)malloc(k * sizeof(int32_t));
  int16_t* expect_sc  = (int16_t*)malloc(k * sizeof(int16_t));
  int32_t* actual_idx = (int32_t*)malloc(k * sizeof(int32_t));
  int16_t* actual_sc  = (int16_t*)malloc(k * sizeof(int16_t));

  brute_force_top_k(scores, N, k, expect_idx, expect_sc);
  top_k_from_scores(scores, N, k, actual_idx, actual_sc);

  int pass = 1;
  for (int32_t i = 0; i < k; i++) {
    if (actual_sc[i] != expect_sc[i]) { pass = 0; break; }
  }

  printf("%s [N=%d, k=%d]: %s\n", name, N, k, pass ? "PASS" : "FAIL");
  if (!pass) {
    printf("  expected: ");
    for (int i = 0; i < k; i++) printf("(%d, %d) ", expect_idx[i], expect_sc[i]);
    printf("\n  actual:   ");
    for (int i = 0; i < k; i++) printf("(%d, %d) ", actual_idx[i], actual_sc[i]);
    printf("\n");
  }

  free(expect_idx); free(expect_sc);
  free(actual_idx); free(actual_sc);
  return pass;
}

int main() {
  int total = 0, passed = 0;

  /* 测试 1: 与 Isabelle 测试相同的数据 */
  {
    int16_t scores[] = {-26672, 6624, 20160, 19520, -27296, -18400, 28320, -3168, -30608, 5440};
    total++; passed += test_case("isabelle_data", scores, 10, 5);
  }

  /* 测试 2: 已排序的升序输入 */
  {
    int16_t scores[] = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10};
    total++; passed += test_case("ascending", scores, 10, 3);
  }

  /* 测试 3: 已排序的降序输入 */
  {
    int16_t scores[] = {10, 9, 8, 7, 6, 5, 4, 3, 2, 1};
    total++; passed += test_case("descending", scores, 10, 3);
  }

  /* 测试 4: 全部相同 */
  {
    int16_t scores[] = {5, 5, 5, 5, 5};
    total++; passed += test_case("all_equal", scores, 5, 3);
  }

  /* 测试 5: k=1 */
  {
    int16_t scores[] = {-100, 200, -300, 400, 50};
    total++; passed += test_case("k=1", scores, 5, 1);
  }

  /* 测试 6: k=N */
  {
    int16_t scores[] = {30, 10, 20};
    total++; passed += test_case("k=N", scores, 3, 3);
  }

  /* 测试 7: 包含 int16 边界值 */
  {
    int16_t scores[] = {32767, -32768, 0, 1, -1};
    total++; passed += test_case("boundary", scores, 5, 3);
  }

  /* 测试 8: 负数为主 */
  {
    int16_t scores[] = {-100, -200, -50, -300, -10, -500};
    total++; passed += test_case("negatives", scores, 6, 3);
  }

  /* 测试 9: 原始 bug 的最小复现 —— sift-up 会丢弃错误元素 */
  {
    int16_t scores[] = {1, 3, 2, 10, 20, 0};
    total++; passed += test_case("original_bug_repro", scores, 6, 5);
  }

  /* 测试 10: 较大 N */
  {
    int16_t scores[200];
    for (int i = 0; i < 200; i++) scores[i] = (int16_t)((i * 137 + 42) % 30000 - 15000);
    total++; passed += test_case("large_N", scores, 200, 10);
  }

  /* 测试 11: k > N 的情况（k 被限制为 min(k,N) 时应正确） */
  {
    int16_t scores[] = {50, 30};
    total++; passed += test_case("k>N", scores, 2, 2);
  }

  /* 测试 12: N=1, k=1 */
  {
    int16_t scores[] = {-42};
    total++; passed += test_case("single", scores, 1, 1);
  }

  /* 测试 13: 大量重复值，少数不同 */
  {
    int16_t scores[50];
    for (int i = 0; i < 50; i++) scores[i] = 100;
    scores[7] = 200;
    scores[33] = 300;
    scores[49] = 150;
    total++; passed += test_case("mostly_equal", scores, 50, 5);
  }

  /* 测试 14: 锯齿形 —— 交替正负 */
  {
    int16_t scores[20];
    for (int i = 0; i < 20; i++) scores[i] = (i % 2 == 0) ? (int16_t)(i * 100) : (int16_t)(-i * 100);
    total++; passed += test_case("zigzag", scores, 20, 5);
  }

  /* 测试 15: 最佳值在最后 */
  {
    int16_t scores[] = {1, 2, 3, 4, 5, 6, 7, 8, 9, 32767};
    total++; passed += test_case("best_last", scores, 10, 3);
  }

  /* 测试 16: 最佳值在最前 */
  {
    int16_t scores[] = {32767, 9, 8, 7, 6, 5, 4, 3, 2, 1};
    total++; passed += test_case("best_first", scores, 10, 3);
  }

  /* 测试 17: 全负数 */
  {
    int16_t scores[] = {-32768, -32767, -1, -2, -100, -30000, -20000, -10000};
    total++; passed += test_case("all_negative", scores, 8, 4);
  }

  /* 测试 18: k 接近 N */
  {
    int16_t scores[] = {10, 50, 30, 20, 40, 60, 70, 80, 90, 100};
    total++; passed += test_case("k_near_N", scores, 10, 8);
  }

  /* 测试 19: 压力测试 N=10000 k=100 */
  {
    int16_t* scores = (int16_t*)malloc(10000 * sizeof(int16_t));
    for (int i = 0; i < 10000; i++)
      scores[i] = (int16_t)(((int64_t)i * 31337 + 12345) % 65536 - 32768);
    total++; passed += test_case("stress_10k", scores, 10000, 100);
    free(scores);
  }

  /* 测试 20: 压力测试 N=50000 k=500 */
  {
    int16_t* scores = (int16_t*)malloc(50000 * sizeof(int16_t));
    for (int i = 0; i < 50000; i++)
      scores[i] = (int16_t)(((int64_t)i * 48271 + 7919) % 65536 - 32768);
    total++; passed += test_case("stress_50k", scores, 50000, 500);
    free(scores);
  }

  printf("\n%d / %d tests passed\n", passed, total);
  return passed == total ? 0 : 1;
}
