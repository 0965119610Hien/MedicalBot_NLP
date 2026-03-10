import json
from underthesea import word_tokenize
from collections import Counter

def build_vocabulary_from_data(json_files):
    """
    Xây dựng từ điển từ dữ liệu thực tế
    """
    print("\n[Xây dựng từ điển] Đọc tất cả dữ liệu...")
    vocabulary = Counter()
    
    for json_file in json_files:
        try:
            with open(json_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            for record in data:
                for key, value in record.items():
                    if isinstance(value, str) and value.strip():
                        # Tách từ
                        words = word_tokenize(value.lower())
                        vocabulary.update(words)
        except Exception as e:
            print(f"  Lỗi khi đọc {json_file}: {e}")
    
    print(f"✓ Đã xây dựng từ điển với {len(vocabulary)} từ duy nhất")
    return vocabulary

def detect_typos(vocabulary, min_frequency=2):
    """
    Phát hiện từ có thể sai chính tả (xuất hiện ít, độ dài ngắn, có ký tự lạ)
    """
    potential_typos = {}
    
    for word, count in vocabulary.items():
        # Bỏ qua từ xuất hiện nhiều (có thể là từ đúng)
        if count < min_frequency:
            # Kiểm tra các dấu hiệu sai chính tả
            if (len(word) <= 2 or  # Từ quá ngắn
                any(char.isdigit() for char in word) or  # Có số
                word.startswith('_') or word.endswith('_')):  # Ký tự đặc biệt
                potential_typos[word] = count
    
    return potential_typos

def normalize_medical_terms(text):
    """
    Chuẩn hóa các thuật ngữ y tế và viết tắt thường gặp
    """
    # Từ điển viết tắt y tế
    abbreviations = {
        'bs': 'bác_sĩ',
        'bs.': 'bác_sĩ',
        'bác sĩ': 'bác_sĩ',
        'bv': 'bệnh_viện',
        'bv.': 'bệnh_viện',
        'bệnh viện': 'bệnh_viện',
        'bn': 'bệnh_nhân',
        'bn.': 'bệnh_nhân',
        'bệnh nhân': 'bệnh_nhân',
        'tp.hcm': 'thành_phố_hồ_chí_minh',
        'tphcm': 'thành_phố_hồ_chí_minh',
        'tp hcm': 'thành_phố_hồ_chí_minh',
        'hn': 'hà_nội',
        'vn': 'việt_nam',
        'xn': 'xét_nghiệm',
        'xn.': 'xét_nghiệm',
        'xét nghiệm': 'xét_nghiệm',
        'y tế': 'y_tế',
        'sức khỏe': 'sức_khỏe',
        'điều trị': 'điều_trị',
        'chẩn đoán': 'chẩn_đoán',
        'triệu chứng': 'triệu_chứng'
    }
    
    for abbrev, full in abbreviations.items():
        text = text.replace(abbrev, full)
    
    return text

def analyze_raw_data(json_file_path):
    """
    Thống kê dữ liệu gốc chưa xử lý
    """
    print(f"\nTHỐNG KÊ DỮ LIỆU GỐC: {json_file_path}")
    print("-" * 80)
    
    with open(json_file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # Thống kê cơ bản
    total_records = len(data)
    print(f"Tổng số bản ghi: {total_records:,}")
    
    # Thống kê độ dài văn bản
    text_lengths = []
    total_chars = 0
    field_counts = Counter()
    
    for record in data:
        record_text = ""
        for key, value in record.items():
            if isinstance(value, str) and value.strip():
                record_text += value + " "
                field_counts[key] += 1
        
        text_length = len(record_text)
        text_lengths.append(text_length)
        total_chars += text_length
    
    print(f"Tổng số ký tự: {total_chars:,}")
    print(f"Trung bình ký tự/bản ghi: {total_chars/total_records:.2f}")
    print(f"Bản ghi ngắn nhất: {min(text_lengths):,} ký tự")
    print(f"Bản ghi dài nhất: {max(text_lengths):,} ký tự")
    
    # Thống kê các trường dữ liệu
    print(f"\nCác trường dữ liệu phổ biến:")
    for i, (field, count) in enumerate(field_counts.most_common(10), 1):
        print(f"  {i}. '{field}': {count:,} bản ghi ({count/total_records*100:.1f}%)")
    
    return {
        'total_records': total_records,
        'total_chars': total_chars,
        'avg_chars': total_chars/total_records,
        'min_chars': min(text_lengths),
        'max_chars': max(text_lengths),
        'field_counts': dict(field_counts)
    }

def analyze_processed_data(processed_data, output_stats_file):
    """
    Thống kê dữ liệu đã xử lý
    """
    print(f"\nTHỐNG KÊ DỮ LIỆU ĐÃ XỬ LÝ")
    print("-" * 80)
    
    # Thu thập tất cả các từ
    all_words = []
    for record in processed_data:
        words = record['cleaned_text'].split()
        all_words.extend(words)
    
    # Thống kê từ
    word_freq = Counter(all_words)
    total_words = len(all_words)
    unique_words = len(word_freq)
    
    print(f"Tổng số từ: {total_words:,}")
    print(f"Số từ duy nhất: {unique_words:,}")
    print(f"Tỷ lệ từ duy nhất: {unique_words/total_words*100:.2f}%")
    
    # Thống kê độ dài từ
    word_lengths = [len(word) for word in all_words]
    avg_word_length = sum(word_lengths) / len(word_lengths) if word_lengths else 0
    print(f"Độ dài trung bình của từ: {avg_word_length:.2f} ký tự")
    
    # Top từ phổ biến
    print(f"\nTop 20 từ phổ biến nhất:")
    for i, (word, count) in enumerate(word_freq.most_common(20), 1):
        print(f"  {i}. '{word}': {count:,} lần ({count/total_words*100:.2f}%)")
    
    # Phân phối tần suất
    freq_distribution = Counter(word_freq.values())
    print(f"\nPhân phối tần suất:")
    print(f"  Từ xuất hiện 1 lần: {freq_distribution[1]:,} từ ({freq_distribution[1]/unique_words*100:.2f}%)")
    print(f"  Từ xuất hiện 2-5 lần: {sum(freq_distribution[i] for i in range(2, 6)):,} từ")
    print(f"  Từ xuất hiện 6-10 lần: {sum(freq_distribution[i] for i in range(6, 11)):,} từ")
    print(f"  Từ xuất hiện >10 lần: {sum(freq_distribution[i] for i in range(11, max(freq_distribution.keys())+1)):,} từ")
    
    # Thống kê theo bản ghi
    word_counts = [r['word_count'] for r in processed_data]
    print(f"\nThống kê theo bản ghi:")
    print(f"  Trung bình từ/bản ghi: {sum(word_counts)/len(word_counts):.2f}")
    print(f"  Bản ghi ít từ nhất: {min(word_counts):,} từ")
    print(f"  Bản ghi nhiều từ nhất: {max(word_counts):,} từ")
    
    # Lưu thống kê chi tiết
    stats = {
        'total_words': total_words,
        'unique_words': unique_words,
        'avg_word_length': avg_word_length,
        'top_50_words': dict(word_freq.most_common(50)),
        'frequency_distribution': dict(freq_distribution),
        'avg_words_per_record': sum(word_counts)/len(word_counts),
        'min_words': min(word_counts),
        'max_words': max(word_counts)
    }
    
    with open(output_stats_file, 'w', encoding='utf-8') as f:
        json.dump(stats, f, ensure_ascii=False, indent=2)
    
    print(f"\n✓ Đã lưu thống kê chi tiết vào '{output_stats_file}'")
    
    return stats

def process_json_data(json_file_path, stopwords_path, output_file_path, vocabulary=None, skip_fields=None):
    """
    Đọc file JSON, gộp tất cả trường văn bản, làm sạch và lưu kết quả
    
    Args:
        json_file_path: Đường dẫn đến file JSON
        stopwords_path: Đường dẫn đến file stop words
        output_file_path: Đường dẫn file output
        vocabulary: Từ điển từ dữ liệu (nếu có)
        skip_fields: Danh sách tên trường cần bỏ qua khi gộp văn bản.
                     Mặc định bỏ qua 'full_text' vì đây là bản tổng hợp
                     của các trường khác (tránh double content).
    """
    if skip_fields is None:
        skip_fields = ['full_text']
    print(f"\n{'='*80}")
    print(f"XỬ LÝ FILE: {json_file_path}")
    print(f"{'='*80}")
    
    # Bước 1: Đọc file JSON
    print("\n[Bước 1] Đọc file JSON...")
    with open(json_file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    print(f"✓ Đã đọc {len(data)} bản ghi")
    
    # Bước 2: Đọc stop words
    print("\n[Bước 2] Đọc danh sách stop words...")
    with open(stopwords_path, 'r', encoding='utf-8') as f:
        stop_words = set(f.read().splitlines())
    print(f"✓ Đã tải {len(stop_words)} stop words")
    
    # Bước 3: Xử lý từng bản ghi
    print("\n[Bước 3] Xử lý từng bản ghi...")
    processed_data = []
    
    for idx, record in enumerate(data):
        # Gộp tất cả các trường văn bản (bỏ qua skip_fields)
        text_fields = []
        for key, value in record.items():
            if key in skip_fields:
                continue
            if isinstance(value, str) and value.strip():
                text_fields.append(value)
        
        # Nối tất cả văn bản lại
        full_text = ' '.join(text_fields)
        
        # Chuyển chữ thường
        text_lower = full_text.lower()
        
        # Chuẩn hóa thuật ngữ y tế
        text_normalized = normalize_medical_terms(text_lower)
        
        # Loại bỏ dấu câu
        punctuation_to_remove = '''!()-[]{};:'"\,<>./?@#$^&*_~'''
        text_no_punct = ''.join(char for char in text_normalized if char not in punctuation_to_remove)
        
        # Tách từ (Word Tokenization)
        words = word_tokenize(text_no_punct)
        
        # Loại bỏ stop words và từ rỗng
        filtered_words = [word for word in words if word not in stop_words and word.strip() and len(word) > 1]
        
        # Lưu kết quả
        processed_record = {
            'id': idx + 1,
            'url': record.get('url', ''),
            'original_text': full_text[:500],  # Chỉ lưu 500 ký tự đầu
            'cleaned_text': ' '.join(filtered_words),
            'word_count': len(filtered_words)
        }
        processed_data.append(processed_record)
        
        if (idx + 1) % 100 == 0:
            print(f"  Đã xử lý {idx + 1}/{len(data)} bản ghi...")
    
    print(f"✓ Hoàn thành xử lý {len(processed_data)} bản ghi")
    
    # Bước 4: Lưu kết quả
    print(f"\n[Bước 4] Lưu kết quả vào {output_file_path}...")
    with open(output_file_path, 'w', encoding='utf-8') as f:
        json.dump(processed_data, f, ensure_ascii=False, indent=2)
    print("✓ Đã lưu file thành công")
    
    # Thống kê
    print(f"\n{'='*80}")
    print("THỐNG KÊ")
    print(f"{'='*80}")
    total_words = sum(r['word_count'] for r in processed_data)
    avg_words = total_words / len(processed_data) if processed_data else 0
    print(f"Tổng số từ: {total_words:,}")
    print(f"Trung bình từ/bản ghi: {avg_words:.2f}")
    if processed_data:
        print(f"Bản ghi ít từ nhất: {min(r['word_count'] for r in processed_data)}")
        print(f"Bản ghi nhiều từ nhất: {max(r['word_count'] for r in processed_data)}")
    
    return processed_data

def main():
    """Hàm chính để chạy chương trình"""
    print("\nBẮT ĐẦU XỬ LÝ DỮ LIỆU...")
    
    # Danh sách các file JSON từ Vinmec
    # Lưu ý: drugs.json có trường 'full_text' là bản tổng hợp của tất cả
    # các trường khác → process_json_data sẽ tự động bỏ qua để tránh double content
    json_files = [
        '../CrawlVinmec/vinmec_complete_data/diseases.json',
        '../CrawlVinmec/vinmec_complete_data/articles.json',
        '../CrawlVinmec/vinmec_complete_data/drug_qa_pairs.json',
        '../CrawlVinmec/vinmec_complete_data/drugs.json',
    ]
    
    # ========== PHẦN 1: THỐNG KÊ DỮ LIỆU GỐC ==========
    print("\n" + "="*80)
    print("PHẦN 1: THỐNG KÊ DỮ LIỆU THU THẬP (DỮ LIỆU GỐC)")
    print("="*80)
    
    raw_stats = {}
    for json_file in json_files:
        stats = analyze_raw_data(json_file)
        file_name = json_file.split('/')[-1]
        raw_stats[file_name] = stats
    
    # Tổng hợp thống kê dữ liệu gốc
    print(f"\n" + "="*80)
    print("TỔNG HỢP DỮ LIỆU GỐC")
    print("="*80)
    total_raw_records = sum(s['total_records'] for s in raw_stats.values())
    total_raw_chars = sum(s['total_chars'] for s in raw_stats.values())
    print(f"Tổng số bản ghi: {total_raw_records:,}")
    print(f"Tổng số ký tự: {total_raw_chars:,}")
    print(f"Trung bình ký tự/bản ghi: {total_raw_chars/total_raw_records:.2f}")
    
    # Lưu thống kê dữ liệu gốc
    with open('raw_data_statistics.json', 'w', encoding='utf-8') as f:
        json.dump(raw_stats, f, ensure_ascii=False, indent=2)
    print("\n✓ Đã lưu thống kê dữ liệu gốc vào 'raw_data_statistics.json'")
    
    # ========== PHẦN 2: XÂY DỰNG TỪ ĐIỂN VÀ PHÁT HIỆN LỖI ==========
    print("\n" + "="*80)
    print("PHẦN 2: PHÂN TÍCH TỪ ĐIỂN VÀ PHÁT HIỆN LỖI CHÍNH TẢ")
    print("="*80)
    vocabulary = build_vocabulary_from_data(json_files)
    
    # Phát hiện từ có thể sai chính tả
    potential_typos = detect_typos(vocabulary, min_frequency=3)
    print(f"\n✓ Phát hiện {len(potential_typos)} từ có thể sai chính tả")
    print("\nTop 20 từ nghi ngờ sai chính tả:")
    for i, (word, count) in enumerate(sorted(potential_typos.items(), key=lambda x: x[1], reverse=True)[:20], 1):
        print(f"  {i}. '{word}' xuất hiện {count} lần")
    
    # Lưu danh sách từ nghi ngờ sai
    with open('potential_typos.json', 'w', encoding='utf-8') as f:
        json.dump(potential_typos, f, ensure_ascii=False, indent=2)
    print("\n✓ Đã lưu danh sách từ nghi ngờ vào 'potential_typos.json'")
    
    # ========== PHẦN 3: XỬ LÝ VÀ LÀM SẠCH DỮ LIỆU ==========
    print("\n" + "="*80)
    print("PHẦN 3: XỬ LÝ VÀ LÀM SẠCH DỮ LIỆU")
    print("="*80)
    
    # Xử lý file diseases.json
    diseases_data = process_json_data(
        json_file_path='../CrawlVinmec/vinmec_complete_data/diseases.json',
        stopwords_path='vietnamese-stopwords.txt',
        output_file_path='cleaned_diseases.json',
        vocabulary=vocabulary
    )
    
    # Xử lý file articles.json
    articles_data = process_json_data(
        json_file_path='../CrawlVinmec/vinmec_complete_data/articles.json',
        stopwords_path='vietnamese-stopwords.txt',
        output_file_path='cleaned_articles.json',
        vocabulary=vocabulary
    )
    
    # Xử lý file drug_qa_pairs.json
    drug_qa_data = process_json_data(
        json_file_path='../CrawlVinmec/vinmec_complete_data/drug_qa_pairs.json',
        stopwords_path='vietnamese-stopwords.txt',
        output_file_path='cleaned_drug_qa_pairs.json',
        vocabulary=vocabulary
    )

    # Xử lý file drugs.json (thông tin thuốc chi tiết)
    # skip_fields mặc định đã bỏ 'full_text' để tránh double content
    drugs_data = process_json_data(
        json_file_path='../CrawlVinmec/vinmec_complete_data/drugs.json',
        stopwords_path='vietnamese-stopwords.txt',
        output_file_path='cleaned_drugs.json',
        vocabulary=vocabulary
    )
    
    # ========== PHẦN 4: THỐNG KÊ DỮ LIỆU ĐÃ XỬ LÝ ==========
    print("\n" + "="*80)
    print("PHẦN 4: THỐNG KÊ DỮ LIỆU ĐÃ XỬ LÝ")
    print("="*80)
    
    # Thống kê từng file
    diseases_stats = analyze_processed_data(diseases_data, 'cleaned_diseases_stats.json')
    articles_stats = analyze_processed_data(articles_data, 'cleaned_articles_stats.json')
    drug_qa_stats = analyze_processed_data(drug_qa_data, 'cleaned_drug_qa_pairs_stats.json')
    drugs_stats = analyze_processed_data(drugs_data, 'cleaned_drugs_stats.json')
    
    # Tổng hợp thống kê
    print(f"\n" + "="*80)
    print("TỔNG HỢP DỮ LIỆU ĐÃ XỬ LÝ")
    print("="*80)
    total_processed_words = (
        diseases_stats['total_words'] + articles_stats['total_words'] +
        drug_qa_stats['total_words'] + drugs_stats['total_words']
    )
    total_unique_words = (
        diseases_stats['unique_words'] + articles_stats['unique_words'] +
        drug_qa_stats['unique_words'] + drugs_stats['unique_words']
    )
    print(f"Tổng số từ đã xử lý: {total_processed_words:,}")
    print(f"Tổng số từ duy nhất: {total_unique_words:,}")
    
    # So sánh trước và sau xử lý
    print(f"\n" + "="*80)
    print("SO SÁNH TRƯỚC VÀ SAU XỬ LÝ")
    print("="*80)
    total_records_processed = len(diseases_data) + len(articles_data) + len(drug_qa_data) + len(drugs_data)
    print(f"Số bản ghi ban đầu: {total_raw_records:,}")
    print(f"Số bản ghi sau xử lý: {total_records_processed:,}")
    print(f"Số ký tự ban đầu: {total_raw_chars:,}")
    print(f"Số từ sau xử lý: {total_processed_words:,}")
    reduction_rate = (1 - total_processed_words / (total_raw_chars/5)) * 100  # Ước tính
    print(f"Tỷ lệ giảm dữ liệu: ~{reduction_rate:.1f}%")
    
    print("\nHOÀN THÀNH TẤT CẢ!")
    print(f"\nĐã tạo các file:")
    print(f"  Dữ liệu đã làm sạch:")
    print(f"     - cleaned_diseases.json")
    print(f"     - cleaned_articles.json")
    print(f"     - cleaned_drug_qa_pairs.json")
    print(f"     - cleaned_drugs.json          ← MỚI: thông tin thuốc chi tiết")
    print(f"  Thống kê:")
    print(f"     - raw_data_statistics.json (thống kê dữ liệu gốc)")
    print(f"     - cleaned_diseases_stats.json")
    print(f"     - cleaned_articles_stats.json")
    print(f"     - cleaned_drug_qa_pairs_stats.json")
    print(f"     - cleaned_drugs_stats.json    ← MỚI")
    print(f"  Phân tích:")
    print(f"     - potential_typos.json (từ nghi ngờ sai chính tả)")

if __name__ == "__main__":
    main()

