import re
import json
import pandas as pd
from transformers import AutoTokenizer

# ==============================
# CONFIG
# ==============================
CLEANED_CSV   = "/kaggle/input/foodhazard/cleaned_aug_data1.csv"    
# CHUNKED_JSON  = "chunked_deberta_512.json"
# TOKENIZER_MODEL_NAME = "microsoft/deberta-v3-large"

CHUNKED_JSON  = "chunked_deberta_512.json"
TOKENIZER_MODEL_NAME = "microsoft/deberta-v3-large"
MAX_TOKENS = 512
CHUNK_OVERLAP = 64
MIN_CHARS = 5
print(">> Loading tokenizer:", TOKENIZER_MODEL_NAME)
tokenizer = AutoTokenizer.from_pretrained(
    TOKENIZER_MODEL_NAME,
    trust_remote_code=True
)
def chunk_by_tokens(text, tokenizer, max_tokens=512, overlap=32):
    """
    Chunk văn bản theo token DeBERTa:
    - Mỗi chunk <= max_tokens
    - Có overlap để giữ ngữ cảnh
    - Không phụ thuộc LangChain / tiktoken
    """
    text = str(text).strip()
    if not text:
        return []

    enc = tokenizer(
        text,
        add_special_tokens=False,
        return_attention_mask=False,
        return_token_type_ids=False
    )["input_ids"]

    chunks = []
    start = 0

    while start < len(enc):
        end = start + max_tokens
        sub_ids = enc[start:end]
        sub_text = tokenizer.decode(sub_ids, skip_special_tokens=True)
        sub_text = re.sub(r"\s+", " ", sub_text).strip()

        if len(sub_text) >= MIN_CHARS:
            chunks.append(sub_text)

        start += max_tokens - overlap

    return chunks

print(">> Loading cleaned CSV:", CLEANED_CSV)
df = pd.read_csv(CLEANED_CSV)

records = []

for idx, row in df.iterrows():

    source_text = str(row.get("merged_text", "") or "").strip()
    if not source_text:
        continue

    chunks = chunk_by_tokens(
        source_text,
        tokenizer,
        max_tokens=MAX_TOKENS,
        overlap=CHUNK_OVERLAP
    )

    if not chunks:
        continue

    stt = int(row.get("stt", idx))

    base_meta = {
        "stt": stt,
        "year": int(row["year"]) if "year" in df.columns and pd.notna(row["year"]) else None,
        "month": int(row["month"]) if "month" in df.columns and pd.notna(row["month"]) else None,
        "day": int(row["day"]) if "day" in df.columns and pd.notna(row["day"]) else None,
        "country": row.get("country", ""),
        "hazard_category": row.get("hazard-category", ""),
        "product_category": row.get("product-category", ""),
        "hazard": row.get("hazard", ""),
        "product": row.get("product", ""),
        "title": row.get("title", ""),
    }

    for j, ch in enumerate(chunks):
        rec = dict(base_meta)
        rec["chunk_id"] = f"{stt}_{j}"
        rec["text"] = ch  # ← dùng huấn luyện classifier
        rec["chunk_text"] = ch
        records.append(rec)

    if (idx + 1) % 100 == 0:
        print(f"Processed {idx+1}/{len(df)} rows")

print(">> Total chunks:", len(records))
with open(CHUNKED_JSON, "w", encoding="utf-8") as f:
    json.dump(records, f, ensure_ascii=False, indent=2)

print(">> Saved JSON:", CHUNKED_JSON)
print("DONE.")

