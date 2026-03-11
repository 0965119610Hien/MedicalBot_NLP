"""
Pipeline làm sạch dữ liệu y tế từ Vinmec và HelloBacSi
=========================================================
Chức năng chính:
  1. Đọc dữ liệu thô từ cả 2 domain (Vinmec + HelloBacSi)
  2. Loại bỏ URL, ký tự đặc biệt, markdown, khoảng trắng thừa
  3. Tách từ dính nhau bằng từ điển tiếng Việt (regex + vocab)
  4. Tokenize tiếng Việt đúng cách bằng underthesea (tách cụm từ)
  5. Loại bỏ stopwords
  6. Gộp tất cả dữ liệu đã làm sạch thành 1 file duy nhất
  7. Xuất thống kê chi tiết

Thư viện cần cài:
  pip install underthesea
"""

import json
import os
import re
from collections import Counter
from underthesea import word_tokenize

# ============================================================
# CẤU HÌNH ĐƯỜNG DẪN
# ============================================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Vinmec
VINMEC_DIR = os.path.join(BASE_DIR, '..', 'CrawlVinmec', 'vinmec_complete_data')
VINMEC_FILES = {
    'articles':      os.path.join(VINMEC_DIR, 'articles.json'),
    'diseases':      os.path.join(VINMEC_DIR, 'diseases.json'),
    'drugs':         os.path.join(VINMEC_DIR, 'drugs.json'),
    'drug_qa_pairs': os.path.join(VINMEC_DIR, 'drug_qa_pairs.json'),
}

# HelloBacSi
HELLOBACSI_DIR = os.path.join(BASE_DIR, '..', 'CrawlHelloBacSi')
HELLOBACSI_FOLDERS = {
    'hellobacsi_data_1': 'articles_1.json',
    'hellobacsi_data_2': 'articles_2.json',
    'hellobacsi_data_3': 'articles_3.json',
    'hellobacsi_data_4': 'articles_4.json',
    'hellobacsi_data_5': 'articles_5.json',
    'hellobacsi_data_7': 'articles_7.json',
    'hellobacsi_data_8': 'articles_8.json',
}

STOPWORDS_PATH = os.path.join(BASE_DIR, 'vietnamese-stopwords.txt')

# Output
OUTPUT_DIR = BASE_DIR
MERGED_OUTPUT = os.path.join(OUTPUT_DIR, 'merged_cleaned_data.json')
MERGED_STATS  = os.path.join(OUTPUT_DIR, 'merged_cleaned_stats.json')

# ============================================================
# TỪ ĐIỂN Y TẾ - dùng để tách từ dính & chuẩn hóa
# ============================================================
MEDICAL_COMPOUND_WORDS = [
    # Cụm 3 từ trở lên
    'tác dụng phụ', 'điều trị bệnh', 'sử dụng thuốc',
    'xét nghiệm máu', 'chẩn đoán bệnh', 'phòng ngừa bệnh',
    'chế độ ăn', 'nguyên nhân gây', 'tăng nguy cơ',
    'phương pháp điều trị', 'tham khảo ý kiến',
    'cổ tử cung', 'đái tháo đường',

    # Cụm 2 từ (y tế)
    'bác sĩ', 'bệnh nhân', 'bệnh viện', 'bệnh lý',
    'điều trị', 'triệu chứng', 'chẩn đoán', 'xét nghiệm',
    'phẫu thuật', 'tác dụng', 'liều dùng', 'liều lượng',
    'dược phẩm', 'dược sĩ', 'y tế', 'sức khỏe',
    'huyết áp', 'tiêm chủng', 'kháng sinh', 'kháng thể',
    'miễn dịch', 'ung thư', 'khối u', 'tế bào',
    'hồng cầu', 'bạch cầu', 'tiểu cầu', 'đường huyết',
    'tim mạch', 'hô hấp', 'tiêu hóa',
    'thần kinh', 'xương khớp', 'da liễu', 'nội tiết',
    'sinh sản', 'thai nhi', 'thai kỳ', 'sơ sinh',
    'vắc xin', 'vi khuẩn',
    'nhiễm trùng', 'viêm nhiễm', 'dị ứng', 'mãn tính',
    'cấp tính', 'biến chứng', 'tái phát', 'di căn',
    'truyền nhiễm', 'lây nhiễm', 'phục hồi', 'chức năng',
    'tổn thương', 'chảy máu', 'xuất huyết', 'phù nề',
    'suy giảm', 'tăng sinh', 'ác tính', 'lành tính',
    'mô bệnh', 'giải phẫu', 'sinh thiết', 'nội soi',
    'siêu âm', 'chụp cắt', 'cộng hưởng', 'phóng xạ',
    'hóa trị', 'xạ trị', 'liệu pháp', 'phác đồ',
    'đề kháng', 'kháng nấm', 'kháng viêm',
    'suy thận', 'suy gan', 'suy tim', 'suy hô hấp',
    'tiểu đường',
    'hệ thống', 'cơ chế', 'tác nhân', 'yếu tố',
    'nguy cơ', 'dấu hiệu', 'chỉ định', 'chống chỉ',
    'thận trọng', 'quá mẫn', 'tương tác', 'hấp thu',
    'chuyển hóa', 'bài tiết', 'nồng độ', 'tác động',
    'cân nhắc', 'theo dõi', 'kiểm tra', 'đánh giá',
    'người bệnh', 'người lớn', 'trẻ em', 'phụ nữ',
    'mang thai', 'cho con bú', 'người già',
]

ABBREVIATION_MAP = {
    'bs': 'bác sĩ',      'bs.': 'bác sĩ',
    'bv': 'bệnh viện',    'bv.': 'bệnh viện',
    'bn': 'bệnh nhân',    'bn.': 'bệnh nhân',
    'xn': 'xét nghiệm',   'xn.': 'xét nghiệm',
    'tp.hcm': 'thành phố hồ chí minh',
    'tphcm': 'thành phố hồ chí minh',
    'tp hcm': 'thành phố hồ chí minh',
    'hn': 'hà nội',
    'vn': 'việt nam',
}

# ============================================================
# HÀM TIỆN ÍCH
# ============================================================

def load_stopwords(path):
    """Đọc danh sách stop words từ file."""
    with open(path, 'r', encoding='utf-8') as f:
        return set(line.strip() for line in f if line.strip())


def remove_urls(text):
    """Loại bỏ tất cả các dạng URL khỏi văn bản."""
    # URL chuẩn
    text = re.sub(r'https?://\S+', '', text)
    # www.xxx
    text = re.sub(r'www\.\S+', '', text)
    # URL bị dính (httpswwwvinmeccomvie...)
    text = re.sub(r'https?[a-z0-9./\-_]+(?:com|vn|org|net)[a-z0-9./\-_]*', '', text, flags=re.IGNORECASE)
    return text


def remove_markdown(text):
    """Loại bỏ cú pháp markdown."""
    text = re.sub(r'#{1,6}\s*', '', text)
    text = re.sub(r'\*{1,2}(.+?)\*{1,2}', r'\1', text)
    text = re.sub(r'\[([^\]]*)\]\([^)]*\)', r'\1', text)
    text = re.sub(r'\[article-yoast-faqs\]', '', text)
    return text


def remove_crawl_metadata(text):
    """Loại bỏ metadata crawl: timestamps, date patterns."""
    text = re.sub(r'\d{4}-\d{2}-\d{2}[T ]\d{2}:\d{2}:\d{2}[\.\d]*', '', text)
    text = re.sub(r'\b\d{8}\b', '', text)
    text = re.sub(r'\b\d{1,2}/\d{1,2}/\d{4}\b', '', text)
    text = re.sub(r'\bt\d{10,}\b', '', text, flags=re.IGNORECASE)
    return text


def normalize_whitespace(text):
    """Chuẩn hóa khoảng trắng."""
    text = text.replace('\n', ' ').replace('\r', ' ').replace('\t', ' ')
    text = re.sub(r'\s+', ' ', text)
    return text.strip()


def remove_special_chars(text):
    """Loại bỏ ký tự đặc biệt, giữ lại chữ, số, khoảng trắng."""
    text = re.sub(r'[^\w\s]', ' ', text, flags=re.UNICODE)
    text = re.sub(r'(?<!\w)_|_(?!\w)', ' ', text)
    return text


def expand_abbreviations(text):
    """Thay thế viết tắt y tế bằng dạng đầy đủ."""
    for abbrev, full in ABBREVIATION_MAP.items():
        pattern = re.compile(r'\b' + re.escape(abbrev) + r'\b', re.IGNORECASE)
        text = pattern.sub(full, text)
    return text


def build_split_patterns():
    """
    Xây dựng regex patterns để tách từ dính nhau.
    Ví dụ: 'điềutrị' → 'điều trị', 'bácsĩ' → 'bác sĩ'
    """
    patterns = []
    for compound in MEDICAL_COMPOUND_WORDS:
        parts = compound.split()
        if len(parts) >= 2:
            joined = ''.join(parts)
            if len(joined) > 3:
                patterns.append((joined, compound))
    # Ưu tiên match dài trước
    patterns.sort(key=lambda x: len(x[0]), reverse=True)
    return patterns


def split_stuck_words(text, patterns):
    """Tách các từ bị dính nhau dựa trên từ điển."""
    for joined, spaced in patterns:
        text = re.sub(re.escape(joined), spaced, text, flags=re.IGNORECASE)
    return text


def tokenize_vietnamese(text):
    """
    Tách từ tiếng Việt bằng underthesea.
    Tự động nhóm cụm từ: 'bác sĩ' → 'bác_sĩ', 'bệnh nhân' → 'bệnh_nhân'
    """
    return word_tokenize(text, format="text")


# ============================================================
# TRÍCH XUẤT VĂN BẢN TỪ CÁC NGUỒN DỮ LIỆU
# ============================================================

def extract_text_vinmec_articles(record):
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


def extract_text_vinmec_diseases(record):
    fields = ['nguyen_nhan', 'yeu_to_nguy_co', 'trieu_chung',
              'chan_doan', 'dieu_tri', 'phong_ngua']
    parts = []
    for field in fields:
        val = record.get(field, '')
        if isinstance(val, str) and val.strip():
            parts.append(val.strip())
    return ' '.join(parts)


def extract_text_vinmec_drugs(record):
    """Bỏ trường full_text để tránh trùng nội dung."""
    fields = ['name', 'formulation', 'drug_group', 'indication',
              'contraindication', 'precaution', 'side_effects',
              'dosage', 'usage_notes']
    parts = []
    for field in fields:
        val = record.get(field, '')
        if isinstance(val, str) and val.strip():
            parts.append(val.strip())
    return ' '.join(parts)


def extract_text_vinmec_drug_qa(record):
    parts = []
    for field in ['question', 'answer']:
        val = record.get(field, '')
        if isinstance(val, str) and val.strip():
            parts.append(val.strip())
    return ' '.join(parts)


def extract_text_hellobacsi(record):
    parts = []
    for field in ['title', 'content']:
        val = record.get(field, '')
        if isinstance(val, str) and val.strip():
            parts.append(val.strip())
    return ' '.join(parts)


# ============================================================
# ĐỌC DỮ LIỆU THÔ
# ============================================================

def load_vinmec_data():
    results = []
    extractors = {
        'articles':      extract_text_vinmec_articles,
        'diseases':      extract_text_vinmec_diseases,
        'drugs':         extract_text_vinmec_drugs,
        'drug_qa_pairs': extract_text_vinmec_drug_qa,
    }
    for source_name, file_path in VINMEC_FILES.items():
        if not os.path.exists(file_path):
            print(f"  [SKIP] Không tìm thấy: {file_path}")
            continue
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        extractor = extractors[source_name]
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


def load_hellobacsi_data():
    results = []
    seen_urls = set()
    total_raw = 0
    total_dup = 0
    for folder, filename in HELLOBACSI_FOLDERS.items():
        filepath = os.path.join(HELLOBACSI_DIR, folder, filename)
        if not os.path.exists(filepath):
            print(f"  [SKIP] Không tìm thấy: {filepath}")
            continue
        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)
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
    print(f"  → Tổng HelloBacSi: {len(results)} bài (loại {total_dup} trùng từ {total_raw} gốc)")
    return results


# ============================================================
# PIPELINE LÀM SẠCH
# ============================================================

def clean_text(raw_text, split_patterns, stop_words):
    """
    Pipeline làm sạch 1 đoạn văn bản:
      1. Loại bỏ URL
      2. Loại bỏ markdown
      3. Loại bỏ metadata crawl (timestamps, IDs)
      4. Chuyển chữ thường
      5. Mở rộng viết tắt
      6. Tách từ dính nhau (dictionary-based)
      7. Loại bỏ ký tự đặc biệt
      8. Chuẩn hóa khoảng trắng
      9. Tokenize tiếng Việt (underthesea) → tách cụm từ
     10. Loại bỏ stopwords + từ quá ngắn + số thuần
    """
    text = remove_urls(raw_text)
    text = remove_markdown(text)
    text = remove_crawl_metadata(text)
    text = text.lower()
    text = expand_abbreviations(text)
    text = split_stuck_words(text, split_patterns)
    text = remove_special_chars(text)
    text = normalize_whitespace(text)
    text = tokenize_vietnamese(text)

    tokens = text.split()
    filtered = []
    for token in tokens:
        t = token.strip()
        if not t:
            continue
        t_check = t.replace('_', ' ')
        if t_check in stop_words or t in stop_words:
            continue
        if len(t) <= 1:
            continue
        if re.match(r'^\d+$', t):
            continue
        filtered.append(t)

    return ' '.join(filtered)


# ============================================================
# HÀM CHÍNH
# ============================================================

def main():
    print("=" * 70)
    print("   PIPELINE LÀM SẠCH DỮ LIỆU Y TẾ (Vinmec + HelloBacSi)")
    print("=" * 70)

    # ── Bước 1: Đọc dữ liệu thô ──
    print("\n[1/5] Đọc dữ liệu thô...")
    vinmec_data = load_vinmec_data()
    hellobacsi_data = load_hellobacsi_data()
    all_raw = vinmec_data + hellobacsi_data
    print(f"\n→ Tổng cộng: {len(all_raw)} bản ghi thô")

    # ── Bước 2: Chuẩn bị công cụ ──
    print("\n[2/5] Chuẩn bị công cụ làm sạch...")
    stop_words = load_stopwords(STOPWORDS_PATH)
    print(f"  ✓ {len(stop_words)} stopwords")
    split_patterns = build_split_patterns()
    print(f"  ✓ {len(split_patterns)} patterns tách từ dính")

    # ── Bước 3: Làm sạch ──
    print("\n[3/5] Làm sạch dữ liệu...")
    cleaned_records = []
    for idx, record in enumerate(all_raw):
        cleaned_text = clean_text(record['raw_text'], split_patterns, stop_words)
        words = cleaned_text.split()
        if len(words) < 3:
            continue
        cleaned_records.append({
            'id': len(cleaned_records) + 1,
            'domain': record['domain'],
            'source': record['source'],
            'cleaned_text': cleaned_text,
            'word_count': len(words),
        })
        if (idx + 1) % 500 == 0:
            print(f"  Đã xử lý {idx + 1}/{len(all_raw)} bản ghi...")

    print(f"  ✓ Hoàn thành: {len(cleaned_records)} bản ghi "
          f"(bỏ {len(all_raw) - len(cleaned_records)} quá ngắn)")

    # ── Bước 4: Lưu file gộp ──
    print("\n[4/5] Lưu dữ liệu đã làm sạch...")
    with open(MERGED_OUTPUT, 'w', encoding='utf-8') as f:
        json.dump(cleaned_records, f, ensure_ascii=False, indent=2)
    print(f"  ✓ Đã lưu: {os.path.basename(MERGED_OUTPUT)}")

    # ── Bước 5: Thống kê ──
    print("\n[5/5] Tính thống kê...")
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
    }

    with open(MERGED_STATS, 'w', encoding='utf-8') as f:
        json.dump(stats, f, ensure_ascii=False, indent=2)
    print(f"  ✓ Đã lưu: {os.path.basename(MERGED_STATS)}")

    # ── In kết quả ──
    print(f"\n{'=' * 70}")
    print("KẾT QUẢ")
    print(f"{'=' * 70}")
    print(f"  Tổng bản ghi     : {len(cleaned_records):,}")
    print(f"  Tổng từ           : {total_words:,}")
    print(f"  Từ duy nhất       : {unique_words:,}")
    print(f"  TB từ/bản ghi     : {stats['avg_words_per_record']}")
    print(f"  Min/Max từ        : {stats['min_words']}/{stats['max_words']}")

    print(f"\n  Theo domain:")
    for d, ds in domain_stats.items():
        print(f"    {d}: {ds['count']:,} bản ghi, {ds['total_words']:,} từ")

    print(f"\n  Theo source:")
    for s, ss in source_stats.items():
        print(f"    {s}: {ss['count']:,} bản ghi, {ss['total_words']:,} từ")

    print(f"\n  Top 20 từ:")
    for i, (word, count) in enumerate(word_freq.most_common(20), 1):
        print(f"    {i:2}. {word:<20s} {count:>8,} lần")

    print(f"\nOutput files:")
    print(f"  → {MERGED_OUTPUT}")
    print(f"  → {MERGED_STATS}")

    # Hiển thị mẫu
    print(f"\n{'=' * 70}")
    print("MẪU DỮ LIỆU ĐÃ LÀM SẠCH (3 bản ghi đầu)")
    print(f"{'=' * 70}")
    for record in cleaned_records[:3]:
        print(f"\n  [ID {record['id']}] ({record['domain']}/{record['source']})")
        text_preview = record['cleaned_text'][:200]
        print(f"  {text_preview}...")
        print(f"  → {record['word_count']} từ")

    print("\n✓ HOÀN THÀNH!")


if __name__ == "__main__":
    main()
