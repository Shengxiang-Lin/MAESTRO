"""Deterministic serialization helpers shared by source-semantic synthesis."""
from hashlib import sha256
import json
def canonical(value):return json.dumps(value,ensure_ascii=False,sort_keys=True,separators=(",",":"))
def digest(value):return sha256(canonical(value).encode("utf-8")).hexdigest()
__all__=["canonical","digest"]
