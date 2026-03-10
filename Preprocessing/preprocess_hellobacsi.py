"""
Tiền xử lý dữ liệu HelloBacSi - Sử dụng cùng pipeline với Vinmec
"""
import json
import os
from underthesea import word_tokenize
from collections import Counter

def normalize_medical_terms(text):
    abbreviations = {
        'bs': 'bác_sĩ', 'bs.': 'bác_sĩ', 'bác sĩ': 'bác_sĩ',
        'bv': 'bệnh_viện', 'bv.': 'bệnh_viện', 'bệnh viện': 'bệnh_viện',
        'bn': 'bệnh_nhân', 'bn.': 'bệnh_nhân', 'bệnh nhân': 'bệnh_nhân',
        'tp.hcm': 'thành_phố_hồ_chí_minh', 'tphcm': 'thành_phố_hồ_chí_minh',
        'tp hcm': 'thành_phố_hồ_chí_minh',
        'hn': 'hà_nội', 'vn': 'việt_nam',
        'xn': 'xét_nghiệm', 'xn.': 'xét_nghiệm', 'xét nghiệm': 'xét_nghiệm',
        'y tế': 'y_tế', 'sức khỏe': 'sức_khỏe',
        'điều trị': 'điều_trị', 'chẩn đoán': 'chẩn_đoán', 'triệu chứng': 'triệu_chứng'
    }
    for abbrev, full in abbreviations.items():
        text = text.replace(abbrev, full)
    return text

def normalize_category(raw_cat):
    """Chuẩn hóa tên danh mục từ slug hoặc tên tiếng Việt thô"""
    slug_map = {
        # URL slugs → tên chuẩn
        'vac-xin': 'Vắc-xin',
        'benh-tim-mach': 'Bệnh tim mạch',
        'di-ung': 'Dị ứng',
        'benh-than-va-duong-tiet-nieu': 'Bệnh thận & Đường tiết niệu',
        'thoi-quen-lanh-manh': 'Thói quen lành mạnh',
        'tam-ly-tam-than': 'Tâm lý - Tâm thần',
        'benh-tai-mui-hong': 'Bệnh tai mũi họng',
        'ho-va-benh-duong-ho-hap': 'Bệnh hô hấp',
        'ung-thu-ung-buou': 'Ung thư - Ung bướu',
        'benh-tieu-hoa': 'Bệnh tiêu hóa',
        'benh-co-xuong-khop': 'Bệnh cơ xương khớp',
        'benh-truyen-nhiem': 'Bệnh truyền nhiễm',
        'benh-ve-mau': 'Bệnh về máu',
        'benh-nao-he-than-kinh': 'Bệnh não & hệ thần kinh',
        'suc-khoe': 'Sức khỏe chung',
        'suc-khoe-phu-nu': 'Sức khỏe phụ nữ',
        'suc-khoe-nam-gioi': 'Sức khỏe nam giới',
        'suc-khoe-tinh-duc': 'Sức khỏe tình dục',
        'mang-thai': 'Mang thai',
        'nuoi-day-con': 'Nuôi dạy con',
        'an-uong-lanh-manh': 'Ăn uống lành mạnh',
        'the-duc-the-thao': 'Thể dục thể thao',
        'tieu-duong-dai-thao-duong': 'Tiểu đường & Đái tháo đường',
        'da-lieu': 'Da liễu',
        'lao-hoa-lanh-manh': 'Lão hóa lành mạnh',
        'thuoc': 'Thuốc',
        'benh': 'Bệnh lý',
        'sharing': 'Chia sẻ',
        'spotlight': 'Nổi bật',
    }
    if not raw_cat:
        return 'Chưa phân loại'
    key = raw_cat.strip().lower()
    if key in slug_map:
        return slug_map[key]
    return raw_cat.strip()


def merge_and_deduplicate_hellobacsi(save_merged=True):
    """
    Gom tất cả dữ liệu HelloBacSi từ các thư mục rải rác về 1 tập hợp,
    loại bỏ trùng lặp theo URL.

    Tại sao cần gom trước khi clean/embed:
    - Cùng 1 URL có thể xuất hiện ở nhiều đợt crawl khác nhau (data_1..data_8)
    - Nếu không dedup, model embedding sẽ học dữ liệu trùng lặp → bias
    - Build vocabulary cần nhìn toàn bộ corpus để thống kê tần suất đúng

    Args:
        save_merged: Nếu True, lưu file hellobacsi_merged_raw.json để tái sử dụng

    Returns:
        (all_articles, raw_stats): list bài viết đã dedup + thống kê từng nguồn
    """
    data_folders = {
        'hellobacsi_data_1': 'articles_1.json',
        'hellobacsi_data_2': 'articles_2.json',
        'hellobacsi_data_3': 'articles_3.json',
        'hellobacsi_data_4': 'articles_4.json',
        'hellobacsi_data_5': 'articles_5.json',
        'hellobacsi_data_7': 'articles_7.json',
        'hellobacsi_data_8': 'articles_8.json',
    }

    seen_urls = set()
    all_articles = []
    raw_stats = {}

    print("\n[Gom dữ liệu] Đọc từ các thư mục crawl...")
    for folder, filename in data_folders.items():
        filepath = os.path.join('..', 'CrawlHelloBacSi', folder, filename)
        if not os.path.exists(filepath):
            print(f"  Bỏ qua (không tìm thấy): {filepath}")
            continue

        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)

        total_chars = 0
        text_lengths = []
        added = 0
        duplicates = 0

        for record in data:
            # Tính thống kê trước khi dedup
            record_text = " ".join(v for v in record.values() if isinstance(v, str) and v.strip())
            text_lengths.append(len(record_text))
            total_chars += len(record_text)

            # Chuẩn hóa category
            record['_source_category'] = normalize_category(record.get('category', ''))

            # Dedup theo URL
            url = record.get('url', '').strip()
            if url and url in seen_urls:
                duplicates += 1
                continue
            if url:
                seen_urls.add(url)
            all_articles.append(record)
            added += 1

        raw_stats[folder] = {
            'total_records': len(data),
            'added_after_dedup': added,
            'duplicates_skipped': duplicates,
            'total_chars': total_chars,
            'avg_chars': total_chars / len(data) if data else 0,
            'min_chars': min(text_lengths) if text_lengths else 0,
            'max_chars': max(text_lengths) if text_lengths else 0,
        }
        print(f"  {folder}: {len(data)} bài → thêm {added} (bỏ {duplicates} trùng)")

    print(f"\n✓ Tổng cộng sau dedup: {len(all_articles)} bài viết (từ {sum(s['total_records'] for s in raw_stats.values())} bản ghi gốc)")

    # Tuỳ chọn: lưu file merged vật lý để tái sử dụng sau này
    if save_merged:
        merged_path = 'hellobacsi_merged_raw.json'
        with open(merged_path, 'w', encoding='utf-8') as f:
            json.dump(all_articles, f, ensure_ascii=False, indent=2)
        print(f"✓ Đã lưu file merged gốc → '{merged_path}' (dùng lại được, không cần đọc lại 7 thư mục)")

    return all_articles, raw_stats


def process_hellobacsi():
    """Xử lý tất cả dữ liệu HelloBacSi từ tất cả các thư mục đã crawl"""

    # Đọc stop words
    with open('vietnamese-stopwords.txt', 'r', encoding='utf-8') as f:
        stop_words = set(f.read().splitlines())
    print(f"Đã tải {len(stop_words)} stop words")

    # Bước 1: Gom + dedup trước khi xử lý
    all_articles, raw_stats = merge_and_deduplicate_hellobacsi(save_merged=True)
    
    # Lưu thống kê dữ liệu gốc
    with open('hellobacsi_raw_stats.json', 'w', encoding='utf-8') as f:
        json.dump(raw_stats, f, ensure_ascii=False, indent=2)
    
    # Tiền xử lý
    print("\nBắt đầu tiền xử lý...")
    processed_data = []
    punctuation_to_remove = '''!()-[]{};:'"\,<>./?@#$^&*_~'''
    
    for idx, record in enumerate(all_articles):
        # Gộp tất cả trường văn bản
        text_fields = []
        for key, value in record.items():
            if isinstance(value, str) and value.strip() and key != '_source_category':
                text_fields.append(value)
        
        full_text = ' '.join(text_fields)
        
        # Chuyển chữ thường
        text_lower = full_text.lower()
        
        # Chuẩn hóa thuật ngữ y tế
        text_normalized = normalize_medical_terms(text_lower)
        
        # Loại bỏ dấu câu
        text_no_punct = ''.join(char for char in text_normalized if char not in punctuation_to_remove)
        
        # Tách từ
        words = word_tokenize(text_no_punct)
        
        # Loại bỏ stop words
        filtered_words = [word for word in words if word not in stop_words and word.strip() and len(word) > 1]
        
        processed_record = {
            'id': idx + 1,
            'url': record.get('url', ''),
            'category': record.get('_source_category', ''),
            'title': record.get('title', ''),
            'original_text': full_text[:500],
            'cleaned_text': ' '.join(filtered_words),
            'word_count': len(filtered_words)
        }
        processed_data.append(processed_record)
        
        if (idx + 1) % 200 == 0:
            print(f"  Đã xử lý {idx + 1}/{len(all_articles)} bản ghi...")
    
    print(f"✓ Hoàn thành xử lý {len(processed_data)} bản ghi")
    
    # Lưu kết quả
    with open('cleaned_hellobacsi_articles.json', 'w', encoding='utf-8') as f:
        json.dump(processed_data, f, ensure_ascii=False, indent=2)
    print("✓ Đã lưu cleaned_hellobacsi_articles.json")
    
    # Thống kê dữ liệu đã xử lý
    all_words = []
    for record in processed_data:
        all_words.extend(record['cleaned_text'].split())
    
    word_freq = Counter(all_words)
    total_words = len(all_words)
    unique_words = len(word_freq)
    word_counts = [r['word_count'] for r in processed_data]
    
    # Thống kê theo category
    cat_stats = {}
    for record in processed_data:
        cat = record['category']
        if cat not in cat_stats:
            cat_stats[cat] = {'count': 0, 'total_words': 0}
        cat_stats[cat]['count'] += 1
        cat_stats[cat]['total_words'] += record['word_count']
    
    stats = {
        'total_records': len(processed_data),
        'total_words': total_words,
        'unique_words': unique_words,
        'avg_words_per_record': sum(word_counts) / len(word_counts) if word_counts else 0,
        'min_words': min(word_counts) if word_counts else 0,
        'max_words': max(word_counts) if word_counts else 0,
        'top_50_words': dict(word_freq.most_common(50)),
        'category_stats': cat_stats
    }
    
    with open('cleaned_hellobacsi_stats.json', 'w', encoding='utf-8') as f:
        json.dump(stats, f, ensure_ascii=False, indent=2)
    
    print(f"\n{'='*60}")
    print(f"THỐNG KÊ HELLOBACSI SAU TIỀN XỬ LÝ")
    print(f"{'='*60}")
    print(f"Tổng bản ghi: {len(processed_data):,}")
    print(f"Tổng số từ: {total_words:,}")
    print(f"Từ duy nhất: {unique_words:,}")
    print(f"TB từ/bản ghi: {stats['avg_words_per_record']:.1f}")
    print(f"Min/Max từ: {stats['min_words']}/{stats['max_words']}")
    print(f"\nTheo danh mục:")
    for cat, cs in cat_stats.items():
        print(f"  {cat}: {cs['count']} bài, {cs['total_words']:,} từ")
    print(f"\nTop 20 từ:")
    for i, (word, count) in enumerate(word_freq.most_common(20), 1):
        print(f"  {i}. '{word}': {count:,}")
    
    return stats

if __name__ == "__main__":
    process_hellobacsi()
