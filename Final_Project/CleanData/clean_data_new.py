"""
Optimized medical text cleaning pipeline.
Tự động đọc dữ liệu từ Vinmec và HelloBacSi.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import unicodedata
from collections import Counter, defaultdict
from dataclasses import dataclass
from typing import Dict, Iterable, List, Sequence, Tuple

from tqdm import tqdm

# ============================================================
# CẤU HÌNH ĐƯỜNG DẪN DỮ LIỆU
# ============================================================
PROJECT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
CLEAN_DATA_DIR = os.path.dirname(__file__)

# Vinmec
VINMEC_DIR = os.path.join(PROJECT_DIR, 'CrawlVinmec', 'vinmec_complete_data')
VINMEC_FILES = {
    'articles':      os.path.join(VINMEC_DIR, 'articles.json'),
    'diseases':      os.path.join(VINMEC_DIR, 'diseases.json'),
    'drugs':         os.path.join(VINMEC_DIR, 'drugs.json'),
    'drug_qa_pairs': os.path.join(VINMEC_DIR, 'drug_qa_pairs.json'),
}

# HelloBacSi
HELLOBACSI_DIR = os.path.join(PROJECT_DIR, 'CrawlHelloBacSi')
HELLOBACSI_FOLDERS = {
    'hellobacsi_data_1': 'articles_1.json',
    'hellobacsi_data_2': 'articles_2.json',
    'hellobacsi_data_3': 'articles_3.json',
    'hellobacsi_data_4': 'articles_4.json',
    'hellobacsi_data_5': 'articles_5.json',
    'hellobacsi_data_7': 'articles_7.json',
    'hellobacsi_data_8': 'articles_8.json',
}

# Support files
STOPWORDS_PATH = os.path.join(CLEAN_DATA_DIR, 'vietnamese-stopwords.txt')
MEDICAL_COMPOUND_WORDS_PATH = os.path.join(CLEAN_DATA_DIR, 'medical_compound_words.txt')
ABBREVIATION_MAP_PATH = os.path.join(CLEAN_DATA_DIR, 'abbreviation_map.txt')

# Output
MERGED_OUTPUT = os.path.join(CLEAN_DATA_DIR, 'merged_cleaned_data.json')
MERGED_STATS = os.path.join(CLEAN_DATA_DIR, 'merged_cleaned_stats.json')

try:
    from underthesea import word_tokenize
except Exception:
    word_tokenize = None

try:
    from datasketch import MinHash, MinHashLSH
    HAS_DATASKETCH = True
except Exception:
    HAS_DATASKETCH = False


# ============================================================
# HÀM TRÍCH XUẤT TEXT TỪ CÁC NGUỒN
# ============================================================

def extract_text_vinmec_articles(record: Dict) -> str:
    """Trích text từ Vinmec articles."""
    parts = []
    for field in ['tieu_de', 'mo_ta']:
        val = record.get(field, '')
        if isinstance(val, str) and val.strip():
            parts.append(val.strip())
    phan_doan = record.get('phan_doan', [])
    if isinstance(phan_doan, list):
        for section in phan_doan:
            if isinstance(section, dict):
                title = section.get('title', '')
                if title and isinstance(title, str):
                    parts.append(title.strip())
                content = section.get('content', [])
                if isinstance(content, list):
                    for c in content:
                        if isinstance(c, str) and c.strip():
                            parts.append(c.strip())
                elif isinstance(content, str) and content.strip():
                    parts.append(content.strip())
    return ' '.join(parts)


def extract_text_vinmec_diseases(record: Dict) -> str:
    """Trích text từ Vinmec diseases."""
    fields = ['nguyen_nhan', 'yeu_to_nguy_co', 'trieu_chung',
              'chan_doan', 'dieu_tri', 'phong_ngua']
    parts = []
    for field in fields:
        val = record.get(field, '')
        if isinstance(val, str) and val.strip():
            parts.append(val.strip())
    return ' '.join(parts)


def extract_text_vinmec_drugs(record: Dict) -> str:
    """Trích text từ Vinmec drugs."""
    fields = ['name', 'formulation', 'drug_group', 'indication',
              'contraindication', 'precaution', 'side_effects',
              'dosage', 'usage_notes']
    parts = []
    for field in fields:
        val = record.get(field, '')
        if isinstance(val, str) and val.strip():
            parts.append(val.strip())
    return ' '.join(parts)


def extract_text_vinmec_drug_qa(record: Dict) -> str:
    """Trích text từ Vinmec drug Q&A."""
    parts = []
    for field in ['question', 'answer']:
        val = record.get(field, '')
        if isinstance(val, str) and val.strip():
            parts.append(val.strip())
    return ' '.join(parts)


def extract_text_hellobacsi(record: Dict) -> str:
    """Trích text từ HelloBacSi articles."""
    parts = []
    for field in ['title', 'content']:
        val = record.get(field, '')
        if isinstance(val, str) and val.strip():
            parts.append(val.strip())
    return ' '.join(parts)


# ============================================================
# HÀM ĐỌC DỮ LIỆU TỪ VINMEC VÀ HELLOBACSI
# ============================================================

def load_vinmec_data() -> List[Dict]:
    """Đọc tất cả dữ liệu Vinmec (articles, diseases, drugs, drug_qa_pairs)."""
    results = []
    extractors = {
        'articles': extract_text_vinmec_articles,
        'diseases': extract_text_vinmec_diseases,
        'drugs': extract_text_vinmec_drugs,
        'drug_qa_pairs': extract_text_vinmec_drug_qa,
    }
    
    print("\n[*] Đang đọc dữ liệu Vinmec...")
    for source_name, file_path in VINMEC_FILES.items():
        if not os.path.exists(file_path):
            print(f"  [SKIP] Không tìm thấy: {file_path}")
            continue
        
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
        except Exception as e:
            print(f"  [ERROR] Lỗi đọc {file_path}: {e}")
            continue
        
        extractor = extractors.get(source_name)
        if not extractor:
            continue
        
        count = 0
        for record in data:
            text = extractor(record)
            if text.strip():
                results.append({
                    'raw_text': text,
                    'domain': 'vinmec',
                    'source': source_name,
                    'url': record.get('url', ''),
                })
                count += 1
        print(f"  ✓ Vinmec/{source_name}: {count} bản ghi")
    
    return results


def load_hellobacsi_data() -> List[Dict]:
    """Đọc tất cả dữ liệu HelloBacSi, tránh duplicate URL."""
    results = []
    seen_urls = set()
    total_raw = 0
    total_dup = 0
    
    print("\n[*] Đang đọc dữ liệu HelloBacSi...")
    for folder, filename in HELLOBACSI_FOLDERS.items():
        filepath = os.path.join(HELLOBACSI_DIR, folder, filename)
        if not os.path.exists(filepath):
            print(f"  [SKIP] Không tìm thấy: {filepath}")
            continue
        
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                data = json.load(f)
        except Exception as e:
            print(f"  [ERROR] Lỗi đọc {filepath}: {e}")
            continue
        
        added = 0
        duplicates = 0
        for record in data:
            total_raw += 1
            url = record.get('url', '').strip()
            if url and url in seen_urls:
                duplicates += 1
                total_dup += 1
                continue
            if url:
                seen_urls.add(url)
            
            text = extract_text_hellobacsi(record)
            if text.strip():
                results.append({
                    'raw_text': text,
                    'domain': 'hellobacsi',
                    'source': folder,
                    'url': url,
                })
                added += 1
        
        print(f"  ✓ HelloBacSi/{folder}: {len(data)} bài → {added} (bỏ {duplicates} trùng)")
    
    print(f"  → Tổng HelloBacSi: {len(results)} bài (loại {total_dup} trùng URL)")
    return results


@dataclass
class CleanerConfig:
    min_words_per_record: int = 10
    enable_exact_dedup: bool = True
    enable_fuzzy_dedup: bool = True
    fuzzy_dedup_threshold: float = 0.92
    fuzzy_bucket_topk: int = 24
    fuzzy_min_words: int = 30
    keep_compounds_with_underscore: bool = True
    normalize_units: bool = True
    protected_units: Tuple[str, ...] = ("mg", "g", "mcg", "ml", "l", "%", "iu", "mmhg")
    enable_pre_chunk_filter: bool = False
    target_records_for_chunking: int = 0
    min_chars_for_chunk_filter: int = 300
    max_chars_for_chunk_filter: int = 50000
    generic_removal_ratio_threshold: float = 0.6


GENERIC_PATTERNS = [
    r"liên hệ.*?(chúng tôi|đội ngũ)",
    r"thông tin.*?(được cung cấp|dành cho)",
    r"mục đích.*?(giáo dục|tham khảo)",
    r"bạn nên.*?(tham khảo|hỏi|gặp)",
    r"để lại bình luận",
    r"chia sẻ.*?bạn bè",
    r"đăng ký.*?nhận tin",
    r"theo dõi.*?chúng tôi",
    r"trang web|website|facebook|instagram",
    r"bài viết.*?liên quan",
    r"copyright|all rights reserved",
    r"(^|\s)(xem thêm|bài viết tiếp|tiếp theo)(\s|$)",
]
GENERIC_REGEXES = [re.compile(p, re.IGNORECASE) for p in GENERIC_PATTERNS]
MEDICAL_KEYWORDS = ("bệnh", "thuốc", "triệu chứng", "điều trị", "chẩn đoán", "hoạt chất", "liều")


def remove_generic_fragments(text: str) -> str:
    cleaned = text
    for pattern in GENERIC_REGEXES:
        cleaned = pattern.sub(" ", cleaned)
    return normalize_whitespace(cleaned)


def score_record_for_chunking(record: Dict) -> int:
    source_priority = {
        "articles": 3,
        "diseases": 2,
        "drugs": 2,
        "drug_qa_pairs": 2,
        "hellobacsi_data_1": 1,
        "hellobacsi_data_3": 1,
        "hellobacsi_data_7": 1,
        "hellobacsi_data_2": 0,
        "hellobacsi_data_4": 0,
        "hellobacsi_data_8": 0,
        "hellobacsi_data_5": -1,
    }
    text = record.get("cleaned_text", "")
    text_len = len(text)
    len_score = 0
    if 1000 <= text_len <= 10000:
        len_score = 2
    elif 500 <= text_len <= 1000 or 10000 < text_len <= 30000:
        len_score = 1

    med_score = sum(text.count(keyword) for keyword in MEDICAL_KEYWORDS)
    med_score = min(med_score // 5, 2)
    return source_priority.get(record.get("source", ""), 0) + len_score + med_score


def filter_records_before_chunking(records: List[Dict], config: CleanerConfig) -> Tuple[List[Dict], Dict[str, int]]:
    stats = Counter()

    length_filtered: List[Dict] = []
    for record in records:
        text = record.get("cleaned_text", "")
        if len(text) < config.min_chars_for_chunk_filter:
            stats["pre_chunk_drop_too_short"] += 1
            continue
        if len(text) > config.max_chars_for_chunk_filter:
            stats["pre_chunk_drop_too_long"] += 1
            continue
        length_filtered.append(record)

    generic_filtered: List[Dict] = []
    for record in length_filtered:
        original_text = record.get("cleaned_text", "")
        cleaned_text = remove_generic_fragments(original_text)
        if not cleaned_text:
            stats["pre_chunk_drop_empty_after_generic"] += 1
            continue
        if len(cleaned_text) < len(original_text) * config.generic_removal_ratio_threshold:
            stats["pre_chunk_drop_generic_heavy"] += 1
            continue
        updated = dict(record)
        updated["cleaned_text"] = cleaned_text
        updated["word_count"] = len(cleaned_text.split())
        generic_filtered.append(updated)

    exact_deduped, exact_removed = deduplicate_exact_records(generic_filtered)
    stats["pre_chunk_exact_dedup_removed"] = exact_removed

    fuzzy_deduped, fuzzy_removed = deduplicate_fuzzy_records(exact_deduped, threshold=0.88)
    stats["pre_chunk_fuzzy_dedup_removed"] = fuzzy_removed

    if config.target_records_for_chunking and len(fuzzy_deduped) > config.target_records_for_chunking:
        ranked = sorted(fuzzy_deduped, key=score_record_for_chunking, reverse=True)
        selected = ranked[:config.target_records_for_chunking]
        stats["pre_chunk_target_selected"] = len(selected)
        stats["pre_chunk_target_dropped"] = len(ranked) - len(selected)
    else:
        selected = fuzzy_deduped
        stats["pre_chunk_target_selected"] = len(selected)
        stats["pre_chunk_target_dropped"] = 0

    for index, record in enumerate(selected, start=1):
        record["id"] = index

    stats["pre_chunk_input_records"] = len(records)
    stats["pre_chunk_output_records"] = len(selected)
    return selected, dict(stats)


# -----------------------------
# File loading helpers
# -----------------------------

def read_key_value_mapping(path: str) -> Dict[str, str]:
    mapping: Dict[str, str] = {}
    if not path or not os.path.exists(path):
        return mapping

    with open(path, "r", encoding="utf-8") as file:
        for line in file:
            item = line.strip()
            if not item or item.startswith("#"):
                continue
            if "=" not in item:
                continue
            key, value = item.split("=", 1)
            key = key.strip().lower()
            value = value.strip().lower()
            if key and value:
                mapping[key] = value
    return mapping


def read_line_list(path: str) -> List[str]:
    items: List[str] = []
    if not path or not os.path.exists(path):
        return items

    with open(path, "r", encoding="utf-8") as file:
        for line in file:
            item = line.strip()
            if not item or item.startswith("#"):
                continue
            items.append(item.lower())
    return items


# -----------------------------
# Text normalization helpers
# -----------------------------

URL_PATTERN = re.compile(r"https?://\S+|www\.\S+", re.IGNORECASE)
MARKDOWN_LINK_PATTERN = re.compile(r"\[([^\]]+)\]\([^)]*\)")
HTML_TAG_PATTERN = re.compile(r"<[^>]+>")
MULTISPACE_PATTERN = re.compile(r"\s+")


def remove_urls_and_markup(text: str) -> str:
    text = URL_PATTERN.sub(" ", text)
    text = MARKDOWN_LINK_PATTERN.sub(r"\1", text)
    text = HTML_TAG_PATTERN.sub(" ", text)
    return text


def normalize_whitespace(text: str) -> str:
    return MULTISPACE_PATTERN.sub(" ", text).strip()


def mask_personal_information(text: str) -> Tuple[str, Dict[str, int]]:
    stats = Counter()
    text, count = re.subn(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}", "<EMAIL>", text)
    stats["emails"] += count
    text, count = re.subn(r"(\+?84|0)(\s|-)?\d{1,3}(\s|-)?\d{3,5}(\s|-)?\d{3,4}", "<PHONE>", text)
    stats["phones"] += count
    return text, dict(stats)


def build_abbreviation_patterns(abbreviation_map: Dict[str, str]) -> List[Tuple[re.Pattern, str]]:
    """Pre-compile all abbreviation patterns for efficiency."""
    patterns = []
    for abbreviation, full_form in abbreviation_map.items():
        try:
            pattern = re.compile(r"\b" + re.escape(abbreviation) + r"\b", re.IGNORECASE)
            patterns.append((pattern, full_form))
        except Exception:
            continue
    return patterns


def expand_abbreviations(text: str, abbreviation_patterns: List[Tuple[re.Pattern, str]]) -> str:
    """Apply pre-compiled abbreviation patterns to text."""
    for pattern, full_form in abbreviation_patterns:
        try:
            text = pattern.sub(full_form, text, count=100)  # Limit substitutions per pattern
        except Exception:
            continue
    return text


def build_compound_phrase_patterns(compound_phrases: Sequence[str]) -> List[Tuple[str, str]]:
    patterns: List[Tuple[str, str]] = []
    for phrase in compound_phrases:
        parts = phrase.split()
        if len(parts) >= 2:
            joined = "".join(parts)
            if len(joined) > 3:
                patterns.append((joined, phrase))
    patterns.sort(key=lambda item: len(item[0]), reverse=True)
    return patterns


def split_joined_compound_words(text: str, patterns: Sequence[Tuple[str, str]]) -> str:
    for joined_form, spaced_form in patterns:
        text = re.sub(re.escape(joined_form), spaced_form, text, flags=re.IGNORECASE)
    return text


def remove_special_characters(text: str) -> str:
    text = re.sub(r"[^\w\s]", " ", text, flags=re.UNICODE)
    text = re.sub(r"(?<!\w)_|_(?!\w)", " ", text)
    return text


def tokenize_vietnamese_text(text: str) -> str:
    if word_tokenize is None:
        return text
    return word_tokenize(text, format="text").replace("_", " ")


def split_text_into_tokens(text: str) -> List[str]:
    if not text:
        return []
    return text.split()


def normalize_number_unit_tokens(tokens: List[str], units: Sequence[str]) -> List[str]:
    if not tokens:
        return tokens

    normalized: List[str] = []
    index = 0
    unit_set = {unit.lower() for unit in units}

    while index < len(tokens):
        current = tokens[index]
        if re.fullmatch(r"\d+(?:\.\d+)?", current) and index + 1 < len(tokens):
            next_token = tokens[index + 1].lower()
            if next_token in unit_set:
                normalized.append(current + next_token)
                index += 2
                continue
        normalized.append(current)
        index += 1

    return normalized


def remove_stopwords_and_short_tokens(tokens: List[str], stopwords: Iterable[str]) -> List[str]:
    stopword_set = set(stopwords)
    filtered: List[str] = []
    for token in tokens:
        item = token.strip()
        if not item:
            continue
        token_for_check = item.replace("_", " ")
        if token_for_check in stopword_set or item in stopword_set:
            continue
        if len(item) <= 1:
            continue
        if re.fullmatch(r"\d+", item):
            continue
        filtered.append(item)
    return filtered


def jaccard_similarity(tokens_a: Sequence[str], tokens_b: Sequence[str]) -> float:
    set_a = set(tokens_a)
    set_b = set(tokens_b)
    if not set_a or not set_b:
        return 0.0
    return len(set_a & set_b) / len(set_a | set_b)


# -----------------------------
# Dedup helpers
# -----------------------------

def create_fuzzy_bucket_signature(tokens: Sequence[str], top_k: int = 24) -> str:
    frequency = Counter(tokens)
    key_tokens = [token for token, _ in sorted(frequency.items(), key=lambda item: (-item[1], item[0]))[:top_k]]
    joined_key = "|".join(key_tokens)
    return hashlib.md5(joined_key.encode("utf-8")).hexdigest()


def deduplicate_exact_records(records: List[Dict], text_field: str = "cleaned_text") -> Tuple[List[Dict], int]:
    unique_records: List[Dict] = []
    seen_hashes = set()
    removed = 0

    for record in records:
        text = record.get(text_field, "")
        record_hash = hashlib.md5(text.encode("utf-8")).hexdigest()
        if record_hash in seen_hashes:
            removed += 1
            continue
        seen_hashes.add(record_hash)
        unique_records.append(record)

    return unique_records, removed


def deduplicate_fuzzy_records(records: List[Dict], threshold: float = 0.92, text_field: str = "cleaned_text") -> Tuple[List[Dict], int]:
    if HAS_DATASKETCH:
        return deduplicate_fuzzy_with_minhash(records, threshold=threshold, text_field=text_field)

    unique_records: List[Dict] = []
    bucket_to_indices: Dict[str, List[int]] = defaultdict(list)
    removed = 0

    for record in records:
        tokens = record.get(text_field, "").split()
        if len(tokens) < 30:
            unique_records.append(record)
            continue

        bucket_signature = create_fuzzy_bucket_signature(tokens)
        candidates = bucket_to_indices.get(bucket_signature, [])
        is_duplicate = False

        for candidate_index in candidates:
            candidate_tokens = unique_records[candidate_index][text_field].split()
            if jaccard_similarity(tokens, candidate_tokens) >= threshold:
                is_duplicate = True
                break

        if is_duplicate:
            removed += 1
            continue

        bucket_to_indices[bucket_signature].append(len(unique_records))
        unique_records.append(record)

    return unique_records, removed


def deduplicate_fuzzy_with_minhash(records: List[Dict], threshold: float = 0.92, text_field: str = "cleaned_text") -> Tuple[List[Dict], int]:
    lsh = MinHashLSH(threshold=threshold, num_perm=128)
    minhashes = {}
    unique_records: List[Dict] = []
    removed = 0

    for index, record in enumerate(records):
        tokens = set(record.get(text_field, "").split())
        if len(tokens) < 30:
            unique_records.append(record)
            continue

        minhash = MinHash(num_perm=128)
        for token in tokens:
            minhash.update(token.encode("utf-8"))

        candidates = lsh.query(minhash)
        if candidates:
            removed += 1
            continue

        lsh.insert(str(index), minhash)
        minhashes[str(index)] = minhash
        unique_records.append(record)

    return unique_records, removed


# -----------------------------
# Chunk/source helpers
# -----------------------------

def combine_text_from_record(record: Dict, fields: Sequence[str]) -> str:
    parts: List[str] = []
    for field in fields:
        value = record.get(field, "")
        if isinstance(value, str) and value.strip():
            parts.append(value.strip())
    return " ".join(parts)


def build_vinmec_article_text(record: Dict) -> str:
    parts = []
    for field in ["tieu_de", "mo_ta"]:
        value = record.get(field, "")
        if isinstance(value, str) and value.strip():
            parts.append(value.strip())

    sections = record.get("phan_doan", [])
    if isinstance(sections, list):
        for section in sections:
            if not isinstance(section, dict):
                continue
            title = section.get("title", "")
            if isinstance(title, str) and title.strip():
                parts.append(title.strip())
            content = section.get("content", [])
            if isinstance(content, list):
                for item in content:
                    if isinstance(item, str) and item.strip():
                        parts.append(item.strip())
            elif isinstance(content, str) and content.strip():
                parts.append(content.strip())
    return " ".join(parts)


def build_vinmec_disease_text(record: Dict) -> str:
    return combine_text_from_record(
        record,
        ["nguyen_nhan", "yeu_to_nguy_co", "trieu_chung", "chan_doan", "dieu_tri", "phong_ngua"],
    )


def build_vinmec_drug_text(record: Dict) -> str:
    return combine_text_from_record(
        record,
        ["name", "formulation", "drug_group", "indication", "contraindication", "precaution", "side_effects", "dosage", "usage_notes"],
    )


def build_vinmec_drug_qa_text(record: Dict) -> str:
    return combine_text_from_record(record, ["question", "answer"])


def build_hellobacsi_text(record: Dict) -> str:
    return combine_text_from_record(record, ["title", "content"])


# -----------------------------
# Pipeline helpers
# -----------------------------

def clean_single_text(
    raw_text: str,
    compound_patterns: Sequence[Tuple[str, str]],
    stopwords: Iterable[str],
    abbreviation_patterns: List[Tuple[re.Pattern, str]],
    config: CleanerConfig,
) -> Tuple[str, int]:
    text = remove_urls_and_markup(raw_text)
    text = text.lower()
    text = expand_abbreviations(text, abbreviation_patterns)
    text = split_joined_compound_words(text, compound_patterns)
    text = remove_special_characters(text)
    text = normalize_whitespace(text)
    text = tokenize_vietnamese_text(text)

    tokens = split_text_into_tokens(text)
    if config.normalize_units:
        tokens = normalize_number_unit_tokens(tokens, config.protected_units)
    tokens = remove_stopwords_and_short_tokens(tokens, stopwords)
    return " ".join(tokens), 0


def process_records_stream(
    input_paths: Sequence[str],
    output_path: str,
    config: CleanerConfig,
    abbreviation_patterns: List[Tuple[re.Pattern, str]],
    compound_phrases: Sequence[str],
    stopwords: Iterable[str],
) -> str:
    compound_patterns = build_compound_phrase_patterns(compound_phrases)
    stats = Counter()
    temp_output = output_path + ".tmp"

    with open(temp_output, "w", encoding="utf-8") as writer:
        for path in input_paths:
            if not os.path.exists(path):
                continue
            with open(path, "r", encoding="utf-8") as reader:
                for line in tqdm(reader, desc=f"processing {os.path.basename(path)}"):
                    try:
                        record = json.loads(line)
                    except Exception:
                        continue

                    raw_text = record.get("text") or record.get("content") or record.get("raw_text") or ""
                    if not raw_text:
                        continue

                    masked_text, pii_stats = mask_personal_information(raw_text)
                    stats.update(pii_stats)
                    cleaned_text, _ = clean_single_text(
                        masked_text,
                        compound_patterns,
                        stopwords,
                        abbreviation_patterns,
                        config,
                    )

                    tokens = cleaned_text.split()
                    if len(tokens) < config.min_words_per_record:
                        stats["dropped_short"] += 1
                        continue

                    record["cleaned_text"] = cleaned_text
                    record["word_count"] = len(tokens)
                    writer.write(json.dumps(record, ensure_ascii=False) + "\n")
                    stats["written"] += 1

    final_records: List[Dict] = []
    with open(temp_output, "r", encoding="utf-8") as reader:
        for line in reader:
            final_records.append(json.loads(line))

    if config.enable_exact_dedup:
        final_records, exact_removed = deduplicate_exact_records(final_records)
        stats["exact_dedup_removed"] = exact_removed

    if config.enable_fuzzy_dedup:
        final_records, fuzzy_removed = deduplicate_fuzzy_records(final_records, threshold=config.fuzzy_dedup_threshold)
        stats["fuzzy_dedup_removed"] = fuzzy_removed

    for index, record in enumerate(final_records, start=1):
        record["id"] = index

    with open(output_path, "w", encoding="utf-8") as writer:
        json.dump(final_records, writer, ensure_ascii=False, indent=2)

    with open(output_path.replace(".json", "_stats.json"), "w", encoding="utf-8") as writer:
        json.dump(dict(stats), writer, ensure_ascii=False, indent=2)

    return output_path


# -----------------------------
# CLI
# -----------------------------

def main() -> None:
    """Main pipeline: tự động đọc dữ liệu Vinmec + HelloBacSi và làm sạch."""
    print("=" * 80)
    print("   PIPELINE LÀM SẠCH DỮ LIỆU Y TẾ (Vinmec + HelloBacSi)")
    print("=" * 80)
    
    # ── Bước 1: Đọc dữ liệu thô ──
    print("\n[1/5] Đang đọc dữ liệu thô...")
    vinmec_data = load_vinmec_data()
    hellobacsi_data = load_hellobacsi_data()
    all_raw = vinmec_data + hellobacsi_data
    print(f"\n→ Tổng cộng: {len(all_raw)} bản ghi thô")
    
    # ── Bước 2: Chuẩn bị công cụ ──
    print("\n[2/5] Chuẩn bị công cụ làm sạch...")
    config = CleanerConfig()
    
    stopwords = read_line_list(STOPWORDS_PATH)
    print(f"  ✓ {len(stopwords)} stopwords")
    
    compound_phrases = read_line_list(MEDICAL_COMPOUND_WORDS_PATH)
    print(f"  ✓ {len(compound_phrases)} medical compounds")
    
    abbreviation_map = read_key_value_mapping(ABBREVIATION_MAP_PATH)
    abbreviation_patterns = build_abbreviation_patterns(abbreviation_map)
    print(f"  ✓ {len(abbreviation_map)} abbreviations ({len(abbreviation_patterns)} patterns)")
    
    compound_patterns = build_compound_phrase_patterns(compound_phrases)
    print(f"  ✓ {len(compound_patterns)} patterns tách từ dính")
    
    # ── Bước 3: Làm sạch ──
    print("\n[3/5] Đang làm sạch dữ liệu...")
    cleaned_records = []
    dropped_short = 0
    pre_chunk_stats = None  # Initialize pre_chunk_stats (will be None if filtering disabled)
    
    for idx, record in enumerate(all_raw):
        cleaned_text, _ = clean_single_text(
            record['raw_text'],
            compound_patterns,
            stopwords,
            abbreviation_patterns,
            config,
        )
        
        words = cleaned_text.split()
        if len(words) < config.min_words_per_record:
            dropped_short += 1
            continue
        
        cleaned_records.append({
            'id': len(cleaned_records) + 1,
            'domain': record['domain'],
            'source': record['source'],
            'url': record.get('url', ''),
            'cleaned_text': cleaned_text,
            'word_count': len(words),
        })
        
        if (idx + 1) % 500 == 0:
            print(f"  Đã xử lý {idx + 1}/{len(all_raw)} bản ghi...")
    
    print(f"  ✓ Hoàn thành clean: {len(cleaned_records)} bản ghi (bỏ {dropped_short} quá ngắn)")
    
    # ── Deduplication ──
    exact_removed = 0
    fuzzy_removed = 0
    
    if config.enable_exact_dedup:
        before = len(cleaned_records)
        cleaned_records, exact_removed = deduplicate_exact_records(cleaned_records)
        print(f"  ✓ Exact dedup: -{exact_removed} bản ghi (còn {len(cleaned_records)})")
    
    if config.enable_fuzzy_dedup:
        before = len(cleaned_records)
        cleaned_records, fuzzy_removed = deduplicate_fuzzy_records(cleaned_records, threshold=config.fuzzy_dedup_threshold)
        print(f"  ✓ Fuzzy dedup: -{fuzzy_removed} bản ghi (còn {len(cleaned_records)})")
    
    # Re-index id
    for i, record in enumerate(cleaned_records, 1):
        record['id'] = i
    
    # ── Bước 5: Lưu dữ liệu ──
    print("\n[5/6] Đang lưu dữ liệu...")
    with open(MERGED_OUTPUT, 'w', encoding='utf-8') as f:
        json.dump(cleaned_records, f, ensure_ascii=False, indent=2)
    print(f"  ✓ Đã lưu: {MERGED_OUTPUT}")
    
    # ── Bước 6: Thống kê ──
    print("\n[6/6] Tính thống kê...")
    all_words = []
    for record in cleaned_records:
        all_words.extend(record['cleaned_text'].split())
    
    word_freq = Counter(all_words)
    total_words = len(all_words)
    unique_words = len(word_freq)
    word_counts = [r['word_count'] for r in cleaned_records]
    
    domain_stats = {}
    for record in cleaned_records:
        d = record['domain']
        if d not in domain_stats:
            domain_stats[d] = {'count': 0, 'total_words': 0}
        domain_stats[d]['count'] += 1
        domain_stats[d]['total_words'] += record['word_count']
    
    source_stats = {}
    for record in cleaned_records:
        s = f"{record['domain']}/{record['source']}"
        if s not in source_stats:
            source_stats[s] = {'count': 0, 'total_words': 0}
        source_stats[s]['count'] += 1
        source_stats[s]['total_words'] += record['word_count']
    
    stats = {
        'total_records': len(cleaned_records),
        'total_words': total_words,
        'unique_words': unique_words,
        'avg_words_per_record': round(sum(word_counts) / len(word_counts), 2) if word_counts else 0,
        'min_words': min(word_counts) if word_counts else 0,
        'max_words': max(word_counts) if word_counts else 0,
        'domain_stats': domain_stats,
        'source_stats': source_stats,
        'top_50_words': dict(word_freq.most_common(50)),
        'preprocessing_audit': {
            'dropped_short_records': dropped_short,
            'exact_dedup_removed': exact_removed,
            'fuzzy_dedup_removed': fuzzy_removed,
            'pre_chunk_filter': pre_chunk_stats,
        },
    }
    
    with open(MERGED_STATS, 'w', encoding='utf-8') as f:
        json.dump(stats, f, ensure_ascii=False, indent=2)
    print(f"  ✓ Đã lưu: {MERGED_STATS}")
    
    # ── In kết quả ──
    print(f"\n{'=' * 80}")
    print("KẾT QUẢ")
    print(f"{'=' * 80}")
    print(f"  Tổng bản ghi     : {len(cleaned_records):,}")
    print(f"  Tổng từ           : {total_words:,}")
    print(f"  Từ duy nhất       : {unique_words:,}")
    print(f"  TB từ/bản ghi     : {stats['avg_words_per_record']}")
    print(f"  Min/Max từ        : {stats['min_words']}/{stats['max_words']}")
    print(f"  Bỏ quá ngắn       : {dropped_short}")
    print(f"  Exact dedup bỏ    : {exact_removed}")
    print(f"  Fuzzy dedup bỏ    : {fuzzy_removed}")
    if pre_chunk_stats:
        print(f"  Pre-chunk output  : {len(cleaned_records)}")
    
    print(f"\n  Theo domain:")
    for d, ds in domain_stats.items():
        print(f"    {d}: {ds['count']:,} bản ghi, {ds['total_words']:,} từ")
    
    print(f"\n  Theo source:")
    for s, ss in sorted(source_stats.items()):
        print(f"    {s}: {ss['count']:,} bản ghi, {ss['total_words']:,} từ")
    
    print(f"\n  Top 20 từ:")
    for i, (word, count) in enumerate(word_freq.most_common(20), 1):
        print(f"    {i:2}. {word:<20s} {count:>8,} lần")
    
    print(f"\nOutput files:")
    print(f"  → {MERGED_OUTPUT}")
    print(f"  → {MERGED_STATS}")
    
    # Hiển thị mẫu
    print(f"\n{'=' * 80}")
    print("MẪU DỮ LIỆU ĐÃ LÀM SẠCH (3 bản ghi đầu)")
    print(f"{'=' * 80}")
    for record in cleaned_records[:3]:
        print(f"\n  [ID {record['id']}] ({record['domain']}/{record['source']})")
        text_preview = record['cleaned_text'][:200]
        print(f"  {text_preview}...")
        print(f"  → {record['word_count']} từ")
    
    print("\n✓ HOÀN THÀNH!")


if __name__ == "__main__":
    main()
