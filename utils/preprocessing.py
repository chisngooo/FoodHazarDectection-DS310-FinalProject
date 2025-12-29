import re
import pandas as pd
from bs4 import BeautifulSoup
import re
import difflib
# ==============================
# 1) HTML EXTRACT
# ==============================
def extract_text_from_html(html_content):
    soup = BeautifulSoup(html_content, 'html.parser')
    text = soup.get_text(separator=' ')
    return text.strip()

# ==============================
# 2) BASIC NORMALIZATION
# ==============================
def basic_clean(text):
    text = text.replace('\xa0', ' ')
    text = re.sub(r'\s+', ' ', text)
    text = text.strip()
    return text

# ==============================
# 3) REMOVE NOISE BLOCKS
# (boilerplate, phone, agency, repeated legal segments)
# ==============================
NOISE_PATTERNS = [
]

def remove_recall_boilerplate(text):
    for p in NOISE_PATTERNS:
        text = re.sub(p, "", text, flags=re.IGNORECASE)
    return text

# ==============================
# 4) DEDUPLICATE SENTENCES
# ==============================
def normalize_sentence(s: str) -> str:
    """Chuẩn hoá câu để so trùng lặp (mạnh hơn bản basic)."""
    s = s.lower()
    # bỏ số điện thoại + số dài
    s = re.sub(r'\d{3,}', ' ', s)
    # bỏ ký tự thừa
    s = re.sub(r'[^a-z0-9\s.,;:!?\-"]+', ' ', s)
    # gom khoảng trắng
    s = re.sub(r'\s+', ' ', s)
    return s.strip()

def advanced_deduplicate_sentences(
    text: str,
    min_chars: int = 10,
    fuzzy_threshold: float = 0.95,
    fuzzy_window: int = 50
) -> str:
    """
    - Xoá câu trùng EXACT (s_norm giống nhau).
    - Xoá câu gần giống (similarity >= fuzzy_threshold).
    - fuzzy_window: chỉ so với N câu giữ lại gần nhất (đỡ nặng).
    """
    # tách câu
    sentences = re.split(r'(?<=[.!?])\s+', text)
    cleaned_sentences = []
    normalized_buffer = []  # lưu norm của những câu đã giữ (theo thứ tự)
    seen_exact = set()

    for s in sentences:
        raw = s.strip()
        if not raw:
            continue

        norm = normalize_sentence(raw)

        # bỏ câu quá ngắn (thường là noise / fragment)
        if len(norm) < min_chars:
            continue

        # 1) Kiểm tra trùng exact
        if norm in seen_exact:
            continue

        is_dup = False

        # 2) Kiểm tra trùng gần giống (fuzzy)
        #    chỉ so với các câu gần nhất để đỡ tốn thời gian
        for prev_norm in normalized_buffer[-fuzzy_window:]:
            sim = difflib.SequenceMatcher(None, norm, prev_norm).ratio()
            if sim >= fuzzy_threshold:
                is_dup = True
                break

        if is_dup:
            continue

        # Nếu qua hết check → giữ lại
        cleaned_sentences.append(raw)
        seen_exact.add(norm)
        normalized_buffer.append(norm)

    return " ".join(cleaned_sentences)
def deduplicate_sentences(text):
    return advanced_deduplicate_sentences(
        text,
        min_chars=10,
        fuzzy_threshold=0.95,
        fuzzy_window=50
    )

# ==============================
# 5) NORMALIZE KEY ENTITIES
# (hazard labels and product names)
# ==============================
def normalize_entities(text):
    # Các bạn có thể mở rộng dictionary theo domain
    replacements = {
        r"listeria spp": "listeria monocytogenes",
        r"soy proteins?": "soybeans",
        r"e\.? coli": "escherichia coli",
    }
    for pat, rep in replacements.items():
        text = re.sub(pat, rep, text, flags=re.IGNORECASE)
    return text

# ==============================
# 6) FULL CLEAN PIPELINE
# ==============================
def clean_foodhazard_text(text):
    text = basic_clean(text)
    text = remove_recall_boilerplate(text)
    text = deduplicate_sentences(text)
    text = normalize_entities(text)
    return text.strip()

# ==============================
# APPLY PIPELINE
# ==============================
df = pd.read_csv("data/train_data/aug1/aug_data1.csv")

df["title"] = df["title"].apply(extract_text_from_html).apply(clean_foodhazard_text)
df["text"]  = df["text"].apply(extract_text_from_html).apply(clean_foodhazard_text)

df["merged_text"] = (df["title"] + " " + df["text"]).str.lower().str.strip()

df.to_csv("data/train_data/aug1/cleaned_aug_data1.csv", index=False)
print("CLEAN PIPELINE DONE.")
