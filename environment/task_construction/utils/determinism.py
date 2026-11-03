"""Stable hashing shared by synthetic values and their public surfaces."""
import hashlib
def stable_int(*parts: object)->int:
    digest=hashlib.sha256("|".join(map(str,parts)).encode("utf-8")).digest(); return int.from_bytes(digest[:8],"big")
