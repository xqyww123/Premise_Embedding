#!/usr/bin/env python3
"""Collect semantic interpretations for an Isabelle theory.

Usage:
  1. Start the Isa-REPL server (with Semantic_Collection_App session):
       repl_server 127.0.0.1:6666 HOL /tmp/repl_outputs -l Semantic_Collection_App

  2. Run this script:
       python collect_semantics.py HOL.List
"""
import argparse
import threading
import time

import msgpack as mp
import Isabelle_RPC_Host
import Isabelle_Semantic_Embedding
from IsaREPL import Client

parser = argparse.ArgumentParser(description="Collect semantic interpretations for an Isabelle theory")
parser.add_argument("theory", help="Theory name to interpret (e.g., HOL.List)")
parser.add_argument("--repl-addr", default="127.0.0.1:6666", help="Isa-REPL server address")
parser.add_argument("--rpc-addr", default="127.0.0.1:27182", help="RPC host address")
parser.add_argument("--session", default="HOL", help="Session qualifier for theory name resolution")
args = parser.parse_args()

logger = Isabelle_RPC_Host.mk_logger_(args.rpc_addr, None)

# Launch RPC host in daemon thread
rpc_thread = threading.Thread(
    target=Isabelle_RPC_Host.launch_server_,
    args=(args.rpc_addr, logger), daemon=True)
rpc_thread.start()
time.sleep(1)

import sys

# Connect to pre-running Isa-REPL server
c = Client(args.repl_addr, args.session, timeout=120)
print("Loading theories...", flush=True)
fullnames = c.load_theory([args.theory, "Semantic_Embedding.Semantic_Collection_App"])
print(f"Loaded: {fullnames}", flush=True)

# Run the App — streams tracing via msgpack SOME/NONE protocol
print("Running app...", flush=True)
c.run_app("Semantic_Store.collect")
mp.pack(args.theory, c.cout)
c.cout.flush()

# Read and print tracing messages until done
c.sock.settimeout(600)  # 10 min timeout for interpretation
has_error = False
try:
    while True:
        raw = c.unpack.unpack()
        # App done marker: ((), ()) from REPL_Server.output cout packUnit ()
        if isinstance(raw, (list, tuple)) and len(raw) == 2:
            msg, err = raw
            if err is not None and err != ():
                err_str = err.decode("utf-8") if isinstance(err, bytes) else str(err)
                print(err_str, file=sys.stderr, flush=True)
                has_error = True
                continue
            if msg is None or msg == ():
                # Check if this is the done marker (both sides are unit)
                if (msg is None or msg == ()) and (err is None or err == ()):
                    break
                continue
            if isinstance(msg, bytes):
                msg = msg.decode("utf-8", errors="replace")
            if isinstance(msg, str):
                if msg.startswith("ERROR:"):
                    print(msg, file=sys.stderr, flush=True)
                    has_error = True
                else:
                    print(msg, flush=True)
        elif raw is None:
            break
        else:
            print(f"[unexpected: {raw!r}]", flush=True)
except Exception as e:
    print(f"Connection error: {e}", file=sys.stderr, flush=True)
    has_error = True

if has_error:
    print("Failed.", file=sys.stderr)
    sys.exit(1)
else:
    print("Done.")
c.close()
