import json
import os

OUTPUT_DIR = os.path.dirname(os.path.abspath(__file__))

# Danh sách file cleaned cần gộp: (path, domain, source_name)
CLEANED_FILES = [
    (os.path.join(OUTPUT_DIR, "cleaned_diseases.json"),          "vinmec", "diseases"),
    (os.path.join(OUTPUT_DIR, "cleaned_articles.json"),          "vinmec", "articles"),
    (os.path.join(OUTPUT_DIR, "cleaned_drug_qa_pairs.json"),     "vinmec", "drug_qa_pairs"),
    (os.path.join(OUTPUT_DIR, "cleaned_drugs.json"),             "vinmec", "drugs"),
    (os.path.join(OUTPUT_DIR, "cleaned_hellobacsi_articles.json"), "hellobacsi", "hellobacsi_articles"),
]

OUTPUT_FILE  = os.path.join(OUTPUT_DIR, "merged_cleaned_data.json")
STATS_FILE   = os.path.join(OUTPUT_DIR, "merged_cleaned_stats.json")


def merge_all():
    merged = []
    global_id = 1

    print("=" * 60)
    print("Gộp tất cả file dữ liệu đã làm sạch")
    print("=" * 60)

    for file_path, domain, source in CLEANED_FILES:
        if not os.path.exists(file_path):
            print(f"  [SKIP] Không tìm thấy: {os.path.basename(file_path)}")
            continue

        with open(file_path, "r", encoding="utf-8") as f:
            records = json.load(f)

        count = 0
        for record in records:
            entry = {
                "id":           global_id,
                "cleaned_text": record.get("cleaned_text", ""),
                "word_count":   record.get("word_count", 0),
            }
            merged.append(entry)
            global_id += 1
            count += 1

        print(f"  ✓ {os.path.basename(file_path):<45} {count:>6} bản ghi")

    print("-" * 60)
    print(f"  Tổng cộng: {len(merged)} bản ghi")

    # Lưu file gộp
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(merged, f, ensure_ascii=False, indent=2)
    print(f"\n✓ Đã lưu: {os.path.basename(OUTPUT_FILE)}")

    # Tính thống kê
    total_words = sum(r["word_count"] for r in merged)

    stats = {
        "total_records": len(merged),
        "total_words":   total_words,
    }

    with open(STATS_FILE, "w", encoding="utf-8") as f:
        json.dump(stats, f, ensure_ascii=False, indent=2)

    print(f"✓ Đã lưu thống kê: {os.path.basename(STATS_FILE)}")
    print("\n--- Thống kê ---")
    print(f"  Tổng bản ghi : {len(merged):,}")
    print(f"  Tổng từ      : {total_words:,}")


if __name__ == "__main__":
    merge_all()
