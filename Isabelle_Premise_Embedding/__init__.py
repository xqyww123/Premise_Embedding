from typing import cast
from Isabelle_RPC_Host import Connection, isabelle_remote_procedure
from rocksdict import Rdict
import rocksdict
import platformdirs
import numpy as np
import os
import re
import time
import requests
import IsaREPL
import re

TEI_BASE_DEFAULT = os.getenv("TEI_BASE", None)
API_KEY_DEFAULT = os.getenv("API_KEY", None)

type config = tuple[
    str | None , # base_url
    str, # MODEL_ID
    str | None, # api_key
    int, # dimension
]
type goal = tuple[
    list[str], # premises
    str, # conclusion
    list[str], # variables
] 
type premise = tuple[
    str, # statement
    list[str], # variables
] 
type vector = bytearray | bytes

@isabelle_remote_procedure("embed")
def embed(arg : tuple[list[bytes | str], config], connection : Connection) -> list[vector]:
    (texts, (base_url, MODEL_ID, api_key, dimension)) = arg
    if connection is not None and getattr(connection.server, "debugging", False) and texts:
        try:
            counts = [_count_tokens(t.decode("utf-8") if isinstance(t, bytes) else t, MODEL_ID) for t in texts]
            p = np.percentile(counts, [25, 50, 75])
            q25, q50, q75 = float(p[0]), float(p[1]), float(p[2])
            n_over_1000 = sum(1 for c in counts if c > 1000)
            n_over_2000 = sum(1 for c in counts if c > 2000)
            n_over_3000 = sum(1 for c in counts if c > 3000)
            token_total = sum(counts)
            connection.server.logger.debug(
                "embed: %d texts, model=%s dim=%s, token_count 25th=%.0f 50th=%.0f 75th=%.0f max=%d total=%d, over_1000=%d over_2000=%d over_3000=%d",
                len(texts), MODEL_ID, dimension, q25, q50, q75, max(counts), token_total, n_over_1000, n_over_2000, n_over_3000,
            )
        except Exception:
            connection.server.logger.debug("embed: %d texts, model=%s dim=%s (token stats unavailable)", len(texts), MODEL_ID, dimension)
    _t0 = time.perf_counter()
    if base_url is None:
        base_url = TEI_BASE_DEFAULT
        api_key = API_KEY_DEFAULT
    if base_url is None and not MODEL_ID.startswith('reasonwang'):
        raise Exception("the environment variable TEI_BASE_DEFAULT is not set. You must indicate the address of the Hugging Face TEI server.")
    if api_key == "":
        api_key = API_KEY_DEFAULT

    cache_dir = platformdirs.user_cache_dir("Isabelle_Premise_Embedding", "Qiyuan")
    cache_file = os.path.join(cache_dir, f"{MODEL_ID.replace('/', '_')}.db")
    with Rdict(cache_file, options=rocksdict.Options(raw_mode=True)) as db:
        byte_num = dimension * 2
        output: list[bytes | None] = [None] * len(texts)
        new_texts : list[bytes] = []
        indexes = []
        for i, text in enumerate(texts):
            if isinstance(text, str):
                text = text.encode('utf-8')
            v = db.get(text, None)
            if v is None:
                new_texts.append(text)
                indexes.append(i)
            else:
                output[i] = v

        if len(new_texts) == 0:
            if connection is not None and getattr(connection.server, "debugging", False):
                connection.server.logger.debug("embed: elapsed=%.3fs", time.perf_counter() - _t0)
            return cast(list[vector], [v for v in output if v is not None])

        if MODEL_ID.startswith('reasonwang'):
            # tricks: random normalized vectors (L2 norm = 1)
            rng = np.random.default_rng()
            vecs = rng.standard_normal((len(new_texts), dimension)).astype(np.float32)
            vecs /= np.linalg.norm(vecs, axis=1, keepdims=True)
            vecs_int16 = (vecs * 32768).astype('<i2')  # Q1.15 format (little-endian int16)
            vecs_q15_bytes = [vecs_int16[i].tobytes() for i in range(len(vecs_int16))]
        else:
            url = base_url.rstrip("/") + "/v1/embeddings"
            headers = {"Content-Type": "application/json"}
            if api_key:
                headers["Authorization"] = f"Bearer {api_key}"
            resp = requests.post(
                url,
                json={
                    "input": [text.decode('utf-8') for text in new_texts],
                    "model": MODEL_ID,
                    "encoding_format": "float",
                },
                headers=headers,
                timeout=60,
            )
            resp.raise_for_status()
            data = resp.json()

            # OpenAI format: data = [{"embedding": [...], "index": i, ...}, ...]
            vecs = np.asarray([item["embedding"] for item in data["data"]], dtype=np.float32)
            vecs /= (np.linalg.norm(vecs, axis=1, keepdims=True) + 0.05) # this "+ 0.05" prevents overflow.
            vecs_int16 = (vecs * 32768).astype('<i2')  # Q1.15 format (little-endian int16)
            vecs_q15_bytes = [vecs_int16[i].tobytes() for i in range(len(vecs_int16))]  # list of bytes, one per row

        for i, text in enumerate(new_texts):
            db[text] = vecs_q15_bytes[i]
        for i, j in enumerate(indexes):
            output[j] = vecs_q15_bytes[i]
        if connection is not None and getattr(connection.server, "debugging", False):
            connection.server.logger.debug("embed: elapsed=%.3fs", time.perf_counter() - _t0)
        if len(new_texts) == len(texts):
            return cast(list[vector], vecs_q15_bytes)
        else:
            return cast(list[vector], [v for v in output if v is not None])

def pretty_unicode(s: str):
    s = IsaREPL.Client.pretty_unicode(s)
    s = re.sub(r'::type\b', '', s)
    s = re.sub(r'__\b', '', s)
    s = re.sub(r'‹', '\"', s)
    s = re.sub(r'›', '\"', s)
    return s

def unicode_of_goal(goal: goal) -> goal:
    premises, concl, vars = goal
    premises = [pretty_unicode(premise) for premise in premises]
    concl = pretty_unicode(concl)
    vars = [pretty_unicode(var) for var in vars]
    return (premises, concl, vars)

def unicode_of_premise(premise: premise) -> premise:
    statement, prem_vars = premise
    statement = pretty_unicode(statement)
    prem_vars = [pretty_unicode(var) for var in prem_vars]
    return (statement, prem_vars)

_tokenizers = {}
def _count_tokens(text: str, model: str) -> int:
    from transformers import AutoTokenizer, PreTrainedTokenizerBase
    global _tokenizers
    if model not in _tokenizers:
        _tokenizers[model] = AutoTokenizer.from_pretrained(model)
    tokenizer: PreTrainedTokenizerBase = _tokenizers[model]
    # Use backend (Rust) tokenizer when available: avoids materializing Python list of IDs
    if hasattr(tokenizer, "backend_tokenizer"):
        enc = tokenizer.backend_tokenizer.encode(text, add_special_tokens=False)  # type: ignore[call-arg]
        return len(enc)
    return len(tokenizer.encode(text, add_special_tokens=False))

def _truncate_to_token_limit(text: str, model: str, token_limit: int) -> str:
    """
    Truncate text to at most token_limit tokens for the given model,
    without raising exceptions.
    """
    from transformers import AutoTokenizer, PreTrainedTokenizerBase
    global _tokenizers
    if model not in _tokenizers:
        _tokenizers[model] = AutoTokenizer.from_pretrained(model)
    tokenizer: PreTrainedTokenizerBase = _tokenizers[model]
    token_ids = tokenizer.encode(text, add_special_tokens=False)
    if len(token_ids) <= token_limit:
        return text
    truncated_ids = token_ids[:token_limit]
    return tokenizer.decode(truncated_ids, skip_special_tokens=True, clean_up_tokenization_spaces=True)

def _shrink_tokens(statement: str, for_goal: bool, model: str, token_limit: int) -> str:
    tok_count = _count_tokens(statement, model)
    if tok_count > token_limit:
        if not for_goal:
            return _truncate_to_token_limit(statement, model, token_limit)
            #raise ValueError(f"tok_count {tok_count} is greater than TOKEN_LIMIT {token_limit}")
        lines = statement.split('\n')
        ret_lines = []
        for i,line in enumerate(lines):
            if line.startswith('###'):
                ret_lines.append(line)
                continue
            if not line:
                ret_lines.append(line)
                continue
            tok_count -= _count_tokens(line, model)
            if tok_count <= token_limit:
                if len(lines) <= 0:
                    return _truncate_to_token_limit(statement, model, token_limit)
                    #raise ValueError(f"tok_count {tok_count} is greater than TOKEN_LIMIT {token_limit}")
                ret_lines.extend(lines[i+1:])
                # if count_tokens("".join(ret_lines)) > TOKEN_LIMIT:
                #     raise ValueError(f"DDDD tok_count {count_tokens(''.join(ret_lines))} is greater than TOKEN_LIMIT {TOKEN_LIMIT} {for_goal}")
                return "\n".join(ret_lines)
        return _truncate_to_token_limit(statement, model, token_limit)
        #raise ValueError(f"tok_count {tok_count} is greater than TOKEN_LIMIT {token_limit}")
    return statement

def _trim_context(statement : str, ctxt: str, for_goal: bool, model: str, token_limit: int) -> str:
    tok_count = _count_tokens(statement, model)
    if tok_count == token_limit:
        return statement
    if tok_count > token_limit:
        return _shrink_tokens(statement, for_goal, model, token_limit)

    ret = []
    segs = ctxt.split('\n\n\n')
    for seg in segs:
        if not seg:
            continue
        if not for_goal and seg.startswith('theory '):
            continue
        if tok_count >= token_limit:
            break
        seg = seg + "\n\n"
        cnt = _count_tokens(seg, model)
        if tok_count + cnt > token_limit:
            break
        ret.append(seg)
        tok_count += cnt
    def m(cmd : str) -> int:
        if cmd.startswith('theory '):
            return -1
        elif cmd.startswith('notation '):
            return 0
        elif cmd.startswith('class '):
            return 1
        elif cmd.startswith('fact '):
            return 3
        else:
            return 2
    k_ret = [(m(cmd), cmd) for cmd in ret]
    k_ret.sort(key=lambda x: x[0])
    ret = [cmd for _, cmd in k_ret]
    # if _count_tokens("".join(ret), model) > token_limit:
    #     raise ValueError(f"AAAAAAAAAAAA tok_count {_count_tokens(''.join(ret), model)} is greater than TOKEN_LIMIT {token_limit}")
    return "".join(ret) + statement

type ctxt = str | None
def encode_goal(goal: goal, ctxt: ctxt, model: str, token_limit: int) -> str:
    premises, concl, vars = unicode_of_goal(goal)
    goal_str = "###Variables\n" + "".join([gv + "\n\n" for gv in vars])\
         + "###Premises\n" + "".join([premise + "\n\n" for premise in premises])\
         + "###Conclusion\n" + concl
    if ctxt is None:
        goal_str = _shrink_tokens(goal_str, True, model, token_limit)
    else:
        ctxt = pretty_unicode(ctxt)
        goal_str = _trim_context(goal_str, ctxt, True, model, token_limit)
    goal_str = "#Goal\n" + goal_str
    num = _count_tokens(goal_str, model)
    if num > token_limit + 3:
        raise ValueError(f"BBBB Goal has {num} tokens, which is greater than TOKEN_LIMIT {token_limit}")
    return goal_str

def encode_premise(premise: premise, pctxt: ctxt, model: str, token_limit: int) -> str:
    statement, prem_vars = unicode_of_premise(premise)
    encode = "###Variables\n" + "".join([pv + "\n\n" for pv in prem_vars])\
           + "###Statement\n" + statement
    if pctxt is None:
        encode = _shrink_tokens(encode, False, model, token_limit)
    else:
        pctxt = pretty_unicode(pctxt)
        encode = _trim_context(encode, pctxt, False, model, token_limit)
    encode = "#Fact\n" + encode
    num = _count_tokens(encode, model)
    if num > token_limit + 3:
        raise ValueError(f"CCCC Goal has {num} tokens, which is greater than TOKEN_LIMIT {token_limit}")
    return encode

@isabelle_remote_procedure("embed_goal")
def embed_goal(arg: tuple[goal, ctxt, config, int], connection : Connection) -> vector:
    _t0 = time.perf_counter()
    (goal, ctxt, cfg, token_limit) = arg
    _, MODEL_ID, _, _ = cfg
    goal_str = encode_goal(goal, ctxt, MODEL_ID, token_limit)
    result = embed(([goal_str], cfg), connection)[0]
    if connection is not None and getattr(connection.server, "debugging", False):
        elapsed = time.perf_counter() - _t0
        n = _count_tokens(goal_str, MODEL_ID)
        connection.server.logger.debug(
            "embed_goal: token_limit=%s, token_count 25th=50th=75th=max=%s total=%d, over_1000=%d over_2000=%d over_3000=%d, elapsed=%.3fs",
            token_limit, n, n, (1 if n > 1000 else 0), (1 if n > 2000 else 0), (1 if n > 3000 else 0), elapsed,
        )
    return result

@isabelle_remote_procedure("embed_premises")
def embed_premises(arg: tuple[list[tuple[premise, ctxt]], config, int], connection : Connection) -> list[vector]:
    _t0 = time.perf_counter()
    (premises, cfg, token_limit) = arg
    _, MODEL_ID, _, _ = cfg
    premises_str = [encode_premise(premise, ctxt, MODEL_ID, token_limit) for premise, ctxt in premises]
    result = embed((premises_str, cfg), connection)
    if connection is not None and getattr(connection.server, "debugging", False) and premises_str:
        elapsed = time.perf_counter() - _t0
        counts = [_count_tokens(s, MODEL_ID) for s in premises_str]
        p = np.percentile(counts, [25, 50, 75])
        q25, q50, q75 = float(p[0]), float(p[1]), float(p[2])
        n_over_1000 = sum(1 for c in counts if c > 1000)
        n_over_2000 = sum(1 for c in counts if c > 2000)
        n_over_3000 = sum(1 for c in counts if c > 3000)
        token_total = sum(counts)
        connection.server.logger.debug(
            "embed_premises: %d premises, token_limit=%s, token_count 25th=%.0f 50th=%.0f 75th=%.0f max=%d total=%d, over_1000=%d over_2000=%d over_3000=%d, elapsed=%.3fs",
            len(premises), token_limit, q25, q50, q75, max(counts), token_total, n_over_1000, n_over_2000, n_over_3000, elapsed,
        )
    return result

@isabelle_remote_procedure("embed_goal_and_premises")
def embed_goal_and_premises(arg: tuple[goal, ctxt, list[tuple[premise, ctxt]], config, int], connection : Connection) -> tuple[bytes, list[bytes]]:
    _t0 = time.perf_counter()
    (goal, gctxt, premises, cfg, token_limit) = arg
    _, MODEL_ID, _, _ = cfg
    goal_str = encode_goal(goal, gctxt, MODEL_ID, token_limit)
    codes : list[str] = [encode_premise(premise, ctxt, MODEL_ID, token_limit) for premise, ctxt in premises]
    codes.append(goal_str)
    vecs = embed((codes, cfg), connection)
    goal_vec = vecs.pop()
    prem_vecs = vecs
    if connection is not None and getattr(connection.server, "debugging", False) and codes:
        elapsed = time.perf_counter() - _t0
        counts = [_count_tokens(s, MODEL_ID) for s in codes]
        q25, q50, q75 = (float(np.percentile(counts, 25)), float(np.percentile(counts, 50)), float(np.percentile(counts, 75)))
        n_over_1000 = sum(1 for c in counts if c > 1000)
        n_over_2000 = sum(1 for c in counts if c > 2000)
        n_over_3000 = sum(1 for c in counts if c > 3000)
        token_total = sum(counts)
        connection.server.logger.debug(
            "embed_goal_and_premises: %d premises+goal, token_limit=%s, token_count 25th=%.0f 50th=%.0f 75th=%.0f max=%d total=%d, over_1000=%d over_2000=%d over_3000=%d, elapsed=%.3fs",
            len(codes), token_limit, q25, q50, q75, max(counts), token_total, n_over_1000, n_over_2000, n_over_3000, elapsed,
        )
    return (goal_vec, prem_vecs)

if __name__ == "__main__":
    embed(([b"hello world", b"how to do retrieval with qwen3?"], "https://api.openai.com", "text-embedding-3-small", None, 1536), None)
