#!/usr/bin/env python3
import sys
import json
from rocksdict import Rdict
import rocksdict
import platformdirs
import os

def dump(model_id: str, target_file: str):
    cache_dir = platformdirs.user_cache_dir("Isabelle_Premise_Embedding", "Qiyuan")
    cache_file = os.path.join(cache_dir, f"{model_id.replace('/', '_')}.db")
    with Rdict(cache_file, options=rocksdict.Options(raw_mode=True)) as db:
        with open(target_file, "w") as f:
            for key in db.keys():
                k = key.decode("utf-8", errors="replace") if isinstance(key, bytes) else str(key)
                f.write(json.dumps(k))
                f.write("\n")

if __name__ == "__main__":
    dump(sys.argv[1], sys.argv[2])