import xxhash
from Isabelle_RPC_Host import Connection, isabelle_remote_procedure


@isabelle_remote_procedure("xxhash128_theory")
def theory_xxhash128(arg, connection: Connection) -> bytes:
    """Compute xxhash128 of a theory file combined with parent theory hashes.

    Args:
        arg: (file_path: str|bytes, parent_hashes: list[bytes])
    Returns:
        16-byte xxhash128 digest
    """
    (file_path, parent_hashes) = arg
    if isinstance(file_path, bytes):
        file_path = file_path.decode("utf-8")
    h = xxhash.xxh128()
    with open(file_path, "rb") as f:
        h.update(f.read())
    for ph in parent_hashes:
        h.update(ph)
    return h.digest()
