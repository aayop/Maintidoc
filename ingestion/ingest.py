import os
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"

import re
import uuid
import shutil
from collections import Counter
from pathlib import Path

import pymupdf  # PyMuPDF
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, PointStruct, VectorParams
from sentence_transformers import SentenceTransformer

RAW_DIR = Path("data/raw_docs")
QDRANT_PATH = "./qdrant_data"
COLLECTION = "maintidoc_text"
MODEL_NAME = "intfloat/multilingual-e5-small"
MAX_CHARS = 900
MIN_CHARS = 60  # chunks shorter than this are dropped (titles, page numbers)

ACCESS = {
    "pompes": ["tech_pompes", "managers"],
    "compresseurs": ["tech_compresseurs", "managers"],
}

PAGE_NUM = re.compile(r"^[\-\s]*\d+[\-\s]*$")


def norm_line(line: str) -> str:
    """Normalize a line so 'Page 74' and 'Page 75' look identical."""
    return re.sub(r"\d+", "#", line.strip().lower())


def find_repeated_lines(pdf) -> set[str]:
    """Lines present on at least 30% of pages are headers/footers."""
    counter = Counter()
    for page in pdf:
        lines = {norm_line(l) for l in page.get_text("text").splitlines() if l.strip()}
        counter.update(lines)
    threshold = max(3, int(0.3 * len(pdf)))
    return {l for l, c in counter.items() if c >= threshold and len(l) < 100}


def page_paragraphs(page, repeated: set[str]) -> list[str]:
    paragraphs = []
    for block in page.get_text("blocks", sort=True):
        if block[6] != 0:  # 0 = text block, 1 = image block
            continue
        lines = [l.strip() for l in block[4].splitlines() if l.strip()]
        lines = [l for l in lines if norm_line(l) not in repeated and not PAGE_NUM.match(l)]
        text = "\n".join(lines).strip()
        if text:
            paragraphs.append(text)
    return paragraphs


def pack(paragraphs: list[str], max_chars: int = MAX_CHARS) -> list[str]:
    chunks, current = [], ""
    for p in paragraphs:
        if len(p) > max_chars:
            if current:
                chunks.append(current)
                current = ""
            for i in range(0, len(p), max_chars):
                chunks.append(p[i:i + max_chars])
        elif len(current) + len(p) + 1 <= max_chars:
            current = f"{current}\n{p}".strip()
        else:
            chunks.append(current)
            current = p
    if current:
        chunks.append(current)
    return [c for c in chunks if len(c) >= MIN_CHARS]


def extract_chunks(pdf_path: Path, equipment_type: str) -> list[dict]:
    doc_id = pdf_path.stem
    items = []
    with pymupdf.open(pdf_path) as pdf:
        repeated = find_repeated_lines(pdf)
        print(f"  {len(repeated)} repeated header/footer patterns removed")
        for page_idx, page in enumerate(pdf, start=1):
            paragraphs = page_paragraphs(page, repeated)
            chunks = pack(paragraphs)
            if not chunks:
                print(f"  [!] {doc_id} page {page_idx}: no usable text (image or scan?)")
                continue
            for i, chunk in enumerate(chunks, start=1):
                items.append({
                    "text": chunk,
                    "document_id": doc_id,
                    "document_version": "v1",
                    "page": page_idx,
                    "element_id": f"p{page_idx}_c{i}",
                    "element_type": "text",
                    "equipment_type": equipment_type,
                    "access_groups": ACCESS.get(equipment_type, ["managers"]),
                })
    return items


def main():
    all_items = []
    for family_dir in sorted(RAW_DIR.iterdir()):
        if not family_dir.is_dir():
            continue
        for pdf_path in sorted(family_dir.glob("*.pdf")):
            print(f"Reading {pdf_path.name} ({family_dir.name})")
            all_items.extend(extract_chunks(pdf_path, family_dir.name))

    print(f"{len(all_items)} chunks to index")
    if not all_items:
        print("No PDF found in data/raw_docs/<family>/")
        return

    model = SentenceTransformer(MODEL_NAME)
    dim = len(model.encode("test"))

    shutil.rmtree(QDRANT_PATH, ignore_errors=True)  # wipe the old index completely
    client = QdrantClient(path=QDRANT_PATH)
    client.create_collection(
        COLLECTION, vectors_config=VectorParams(size=dim, distance=Distance.COSINE)
    )

    vectors = model.encode(
        [f"passage: {it['text']}" for it in all_items],
        batch_size=8, show_progress_bar=True, normalize_embeddings=True,
    )
    points = [
        PointStruct(id=str(uuid.uuid4()), vector=vec.tolist(), payload=item)
        for item, vec in zip(all_items, vectors)
    ]
    client.upsert(COLLECTION, points)
    print("Points in index:", client.count(COLLECTION).count, "| expected:", len(all_items))
    client.close()
    print("Done.")

if __name__ == "__main__":
    main()