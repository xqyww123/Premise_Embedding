from Isabelle_RPC_Host import Connection, Remote_Procedures
import numpy as np
import os
import requests

TEI_BASE_DEFAULT = os.getenv("TEI_BASE", None)
API_KEY_DEFAULT = os.getenv("API_KEY", None)

def embed(arg, connection : Connection):
    (texts, base_url, MODEL_ID, api_key) = arg
    if base_url is None:
        base_url = TEI_BASE_DEFAULT
        api_key = API_KEY_DEFAULT
    if base_url is None:
        raise Exception("the environment variable TEI_BASE_DEFAULT is not set. You must indicate the address of the Hugging Face TEI server.")
    if api_key == "":
        api_key = API_KEY_DEFAULT
    url = base_url.rstrip("/") + "/v1/embeddings"
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    resp = requests.post(
        url,
        json={
            "input": texts,
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
    vecs_q15_bytes = (vecs * 32768).astype('<i2').tobytes()  # Convert to Q1.15 format (little-endian int16) and then to byte array
    return vecs_q15_bytes

Remote_Procedures["embed"] = embed



#from sentence_transformers import SentenceTransformer, models

# base = "Qwen/Qwen3-0.6B"  # or any Qwen3 causal LM checkpoint
# 
# # This loads the base model (hidden states) via HF under the hood
# word = models.Transformer(
#     base,
#     model_args={"torch_dtype": "auto"},
#     tokenizer_args={"padding_side": "left"},
# )
# 
# # Decoder-only models usually don't have a CLS token; last-token pooling is common.
# pool = models.Pooling(
#     word.get_word_embedding_dimension(),
#     pooling_mode="lasttoken",   # or "mean"
# )
# 
# embedder = SentenceTransformer(modules=[word, pool])
# 
# vecs = embedder.encode(
#     ["hello world", "how to do retrieval with qwen3?"],
#     normalize_embeddings=True,
# )
# print(vecs.shape)
# 
# # Calculate L2 norm of vecs (original float vectors) before manual normalization
# l2_norms_vecs_before = np.linalg.norm(vecs, axis=1)  # L2 norm along axis 1 (each vector)
# print(f"L2 norms of vecs (before manual normalization): {l2_norms_vecs_before}")
# 
# # Manually normalize to ensure L2 norm is exactly 1.0
# # Note: normalize_embeddings=True may have numerical errors, so we re-normalize explicitly
# l2_norms_vecs = np.linalg.norm(vecs, axis=1, keepdims=True)  # Keep dimensions for broadcasting
# vecs = vecs / l2_norms_vecs  # Normalize each vector to have L2 norm = 1.0
# l2_norms_vecs_after = np.linalg.norm(vecs, axis=1)  # Verify normalization
# print(f"L2 norms of vecs (after manual normalization): {l2_norms_vecs_after}")
# print(f"L2 norms shape: {l2_norms_vecs_after.shape}")  # Should be (2,)
# 
# # Convert to Q1.15 format (16-bit fixed point: 1 integer bit, 15 fractional bits)
# # Q1.15 range: -1.0 to 0.999969482421875
# # Convert float to Q1.15: multiply by 2^15 = 32768, then cast to int16
# # Note: astype(int) truncates towards zero by default (faster than np.trunc())
# #       For positive: floor behavior; for negative: ceil behavior
# vecs_q15 = np.clip(vecs, -1.0, 0.999969482421875)  # Clip to Q1.15 range
# vecs_q15 = (vecs_q15 * 32768).astype(np.int16)  # Truncate towards zero (default behavior), convert to Q1.15 format
# print(f"Q1.15 shape: {vecs_q15.shape}")
# print(f"Q1.15 dtype: {vecs_q15.dtype}")
# print(f"Q1.15 sample values: {vecs_q15[0, :5]}")  # Print first 5 values of first vector
# 
# # Calculate L2 norm of vecs_q15
# # Understanding "axis 1":
# # vecs_q15 shape is (2, 1024):
# #   - axis 0: first dimension (2 vectors) - rows
# #   - axis 1: second dimension (1024 dimensions per vector) - columns
# # 
# # When computing norm along axis=1:
# #   - We compute the norm for each row (each vector)
# #   - Result shape: (2,) - one norm value per vector
# # 
# # Example visualization:
# #   vecs_q15 = [[v1_dim1, v1_dim2, ..., v1_dim1024],  <- vector 1 (row 0)
# #               [v2_dim1, v2_dim2, ..., v2_dim1024]]  <- vector 2 (row 1)
# #               
# #   axis=0 (down)        axis=1 (across) ->
# #   
# #   norm along axis=1: [norm(v1), norm(v2)]  <- one value per row
# 
# # Method 1: Convert back to float and compute L2 norm
# vecs_float = vecs_q15.astype(np.float32) / 32768.0  # Convert Q1.15 back to float
# l2_norms_float = np.linalg.norm(vecs_float, axis=1)  # L2 norm along axis 1 (each vector)
# print(f"L2 norms (converted to float): {l2_norms_float}")
# print(f"L2 norms shape: {l2_norms_float.shape}")  # Should be (2,)
# 
# # Method 2: Compute L2 norm in fixed-point domain (more accurate, avoids float conversion)
# # Note: squares may overflow int16, so use int32 for intermediate calculations
# vecs_q15_int32 = vecs_q15.astype(np.int32)  # Convert to int32 to avoid overflow
# squares = vecs_q15_int32 * vecs_q15_int32  # Square each element
# sum_squares = np.sum(squares, axis=1)  # Sum of squares for each vector
# # Convert sum_squares back to float and divide by (2^15)^2 = 2^30 to get squared norm
# squared_norms = sum_squares.astype(np.float64) / (32768.0 * 32768.0)
# l2_norms_fixed = np.sqrt(squared_norms)  # Take square root to get L2 norm
# print(f"L2 norms (fixed-point computation): {l2_norms_fixed}")
# 
# exit (0)
