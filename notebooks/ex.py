import json
from typing import Any, Dict, List, Optional

def norm(s: Optional[str]) -> str:
    return (s or "").replace("\r", "").strip()

def join_path(parts: List[str]) -> str:
    return " > ".join([p for p in parts if p])

def flatten_legal_tree(node: Dict[str, Any], path: List[str], out: List[Dict[str, Any]]):
    ntype = node.get("type", "")
    title = norm(node.get("title"))
    content = norm(node.get("content"))

    new_path = path + ([title] if title else [])
    if content:
        out.append({"type": ntype, "path": join_path(new_path), "text": content})

    for ch in node.get("children", []) or []:
        flatten_legal_tree(ch, new_path, out)

def load_legal_json(json_path: str) -> Dict[str, Any]:
    with open(json_path, "r", encoding="utf-8") as f:
        return json.load(f)

def flatten_to_passages(doc: Dict[str, Any]) -> List[str]:
    info = doc.get("document_info", {})
    base = [
        f"Văn bản: {norm(info.get('title'))}",
        f"URL: {norm(info.get('url'))}",
        f"Crawled: {norm(info.get('crawled_at'))}",
    ]

    units: List[Dict[str, Any]] = []
    for top in doc.get("body", []) or []:
        flatten_legal_tree(top, base, units)

    passages = []
    for u in units:
        header = f"[{u['type']}] {u['path']}"
        passages.append(header + "\n" + u["text"])
    return passages

JSON_PATH = "thongtu80.json"  # đổi path
doc = load_legal_json(JSON_PATH)
passages = flatten_to_passages(doc)

print("passages:", len(passages))
print(passages[0][:800])










class TokenSplitter:
    def __init__(self, chunk_tokens=800, overlap_tokens=80, tokenizer_name="cl100k_base"):
        self.enc = tiktoken.get_encoding(tokenizer_name)
        self.chunk_tokens = chunk_tokens
        self.overlap_tokens = overlap_tokens

    def chunk(self, text: str):
        toks = self.enc.encode(text)
        step = max(1, self.chunk_tokens - self.overlap_tokens)
        out = []
        for i in range(0, len(toks), step):
            sub = toks[i:i+self.chunk_tokens]
            if sub:
                out.append(self.enc.decode(sub))
        return out

splitter = TokenSplitter(chunk_tokens=800, overlap_tokens=80)

dataset = []
for p in passages:
    dataset.extend(splitter.chunk(p))

print("chunks:", len(dataset))
print(dataset[0][:600])



import numpy as np
import faiss
from sentence_transformers import SentenceTransformer

embed_model_name = "intfloat/multilingual-e5-base"  # hoặc multilingual-e5-large
embedder = SentenceTransformer(embed_model_name)

def embed_texts(texts, batch_size=32):
    # E5: prefix passage:
    inputs = [f"passage: {t}" for t in texts]
    vecs = embedder.encode(inputs, batch_size=batch_size, normalize_embeddings=True)
    return np.asarray(vecs, dtype="float32")

embeddings = embed_texts(dataset, batch_size=32)
print("embeddings:", embeddings.shape)

d = embeddings.shape[1]
index = faiss.IndexFlatIP(d)
index.add(embeddings)
print("FAISS added:", index.ntotal)