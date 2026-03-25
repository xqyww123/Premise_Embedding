# TODO

- [ ] Write a CPython C extension that wraps LMDB directly (using the C `lmdb.h` API), replacing all py-lmdb usage. Expose the required interfaces to Python (open/close env, get/put, txn management, `contains`, `topk`). The `topk` function should: open a read txn, `mdb_get` each candidate key, zero-copy compute dot products against the query vector, and return the top-k results — eliminating the Python loop, `np.frombuffer`, and matrix copy overhead that currently dominates `topk` latency (~0.3s for 16K vectors).

- [ ] Cache candidate entity keys to speed up `candidate listing` (~0.36s for 21K keys, dominated by `key_of_theorem'` at 0.7s on ML side). Entities of persistent (dumped/saved) theories are stable — their universal keys can be cached on the ML side (keyed by theory hash) and reused across queries, avoiding repeated `Term_Digest.thm128` computation.
