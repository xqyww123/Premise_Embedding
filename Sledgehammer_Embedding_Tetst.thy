theory Sledgehammer_Embedding_Tetst
  imports Sledgehammer_Embedding
begin

ML \<open>
val v1 = Word8Vector.tabulate (4096, (fn i => Word8.fromLargeInt (if i mod 2 = 1 then i div 2 else 0)))
val v2 = Word8Vector.tabulate (4096, (fn i => Word8.fromLargeInt (if i mod 2 = 1 then i div 2 else 0)))
\<close>

ML \<open>getenv "LD_LIBRARY_PATH"\<close>

ML \<open>
Vector_Arith_Q15_D2048.dot(v1,v2)
\<close>

ML \<open>
let val vec_size = 4096
    fun rand_vec () = Word8Vector.tabulate (vec_size, fn _ =>
          Word8.fromInt (Random.random_range 0 255))

    fun run_test (trial, (n, k)) =
      let val vecs = List.tabulate (n, fn _ => rand_vec ())
          val query = rand_vec ()

          (* Ground truth: dot 函数计算所有点积 *)
          val dots = map_index (fn (i, v) =>
              (i, Vector_Arith_Q15_D2048.dot (v, query))) vecs
          val sorted = sort (fn ((_,a), (_,b)) => Int.compare (b, a)) dots
          val expected_scores = map snd (take k sorted)

          (* top_k 结果 *)
          val arr = Array.tabulate (n, fn i => (i, nth vecs i))
          val result = Vector_Arith_Q15_D2048.top_k (arr, query, k)
          val actual_scores = map (fn (_, sc) => Real.round (sc * 32768.0)) result

          val pass = (expected_scores = actual_scores)
       in if pass then ()
          else ( writeln (String.concat [
                   "FAIL trial=", string_of_int trial,
                   " n=", string_of_int n, " k=", string_of_int k])
               ; writeln ("  expected: " ^ String.concatWith ","
                   (map string_of_int expected_scores))
               ; writeln ("  actual:   " ^ String.concatWith ","
                   (map string_of_int actual_scores)))
        ; pass
      end

    val configs = [(5,2), (10,5), (50,10), (100,20), (500,50), (1000,100)]
    val num_trials = 10
    val tests = maps (fn cfg => List.tabulate (num_trials, fn t => (t, cfg))) configs
    val results = map run_test tests
    val total = length results
    val passed = length (filter I results)
 in writeln (String.concat [
      "top_k randomized test: ", string_of_int passed,
      " / ", string_of_int total, " passed"])
end
\<close>

end