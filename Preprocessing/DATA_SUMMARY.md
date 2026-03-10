# Tóm tắt các file dữ liệu sau tiền xử lý

> Cập nhật: 09/03/2026  
> Pipeline: Thu thập (crawl) → Làm sạch (clean) → Sẵn sàng embedding

---

## Tổng quan pipeline dữ liệu

```
Nguồn crawl                       Script xử lý                  File output (Preprocessing/)
─────────────────────────────────────────────────────────────────────────────────────────────
CrawlVinmec/vinmec_complete_data/ ──→ clean_data_json.py     ──→ cleaned_diseases.json
                                                              ──→ cleaned_articles.json
                                                              ──→ cleaned_drug_qa_pairs.json
                                                              ──→ cleaned_drugs.json

CrawlHelloBacSi/hellobacsi_data_*/ ─→ preprocess_hellobacsi.py ─→ cleaned_hellobacsi_articles.json
                                                                ──→ hellobacsi_merged_raw.json (tuỳ chọn)
```

---

## 1. File dữ liệu đã làm sạch (Cleaned Data)

### 1.1 `cleaned_diseases.json`

| Thuộc tính | Giá trị |
|---|---|
| **Nguồn gốc** | `CrawlVinmec/vinmec_complete_data/diseases.json` |
| **Script xử lý** | `clean_data_json.py` |
| **Số bản ghi gốc** | 386 bản ghi |
| **Số bản ghi sau xử lý** | 386 bản ghi |
| **Tổng số từ** | 1,470,882 từ |
| **Từ duy nhất** | 7,108 từ |

**Mục đích:** Chứa thông tin các bệnh lý được thu thập từ Vinmec. Dùng để train/embed kiến thức y tế về bệnh học.

**Dữ liệu gốc có các trường:** `url`, `crawl_time`, `nguyen_nhan`, `yeu_to_nguy_co`, `trieu_chung`, `chan_doan`, `dieu_tri`, `phong_ngua`, `noi_dung_day_du`, `bien_chung`, `tien_luong`

**Cấu trúc mỗi bản ghi sau xử lý:**
```json
{
  "id": 1,
  "url": "https://www.vinmec.com/vie/benh/...",
  "original_text": "500 ký tự đầu của văn bản gốc (gộp tất cả các trường)",
  "cleaned_text": "văn bản đã lowercase, bỏ stopwords, word_tokenize (underthesea)",
  "word_count": 2332
}
```

---

### 1.2 `cleaned_articles.json`

| Thuộc tính | Giá trị |
|---|---|
| **Nguồn gốc** | `CrawlVinmec/vinmec_complete_data/articles.json` |
| **Script xử lý** | `clean_data_json.py` |
| **Số bản ghi gốc** | 2,215 bản ghi |
| **Số bản ghi sau xử lý** | 2,215 bản ghi |
| **Tổng số từ** | 3,477,588 từ |
| **Từ duy nhất** | 45,320 từ |

**Mục đích:** Chứa bài viết y tế tổng hợp từ Vinmec (hướng dẫn dùng thuốc, thông tin sức khỏe, giải thích y khoa). Nguồn kiến thức y tế phong phú nhất trong dataset Vinmec.

**Dữ liệu gốc có các trường:** `url`, `crawl_time`, `tieu_de`, `mo_ta`, `noi_dung`

**Cấu trúc mỗi bản ghi sau xử lý:**
```json
{
  "id": 1,
  "url": "https://www.vinmec.com/vie/bai-viet/...",
  "original_text": "500 ký tự đầu...",
  "cleaned_text": "văn bản đã làm sạch",
  "word_count": 530
}
```

---

### 1.3 `cleaned_drug_qa_pairs.json`

| Thuộc tính | Giá trị |
|---|---|
| **Nguồn gốc** | `CrawlVinmec/vinmec_complete_data/drug_qa_pairs.json` |
| **Script xử lý** | `clean_data_json.py` |
| **Số bản ghi gốc** | 546 cặp hỏi-đáp |
| **Số bản ghi sau xử lý** | 546 bản ghi |
| **Tổng số từ** | 24,674 từ |
| **Từ duy nhất** | 1,683 từ |

**Mục đích:** Chứa các cặp câu hỏi–trả lời về thuốc (tên thuốc, chỉ định, liều dùng). Phù hợp để fine-tune QA model hoặc đánh giá retrieval.

**Dữ liệu gốc có các trường:** `question`, `answer`

**Cấu trúc mỗi bản ghi sau xử lý:**
```json
{
  "id": 1,
  "url": "",
  "original_text": "Thuốc Abobotulinum Toxin A dùng để gì? Chỉ định: ...",
  "cleaned_text": "văn bản đã làm sạch",
  "word_count": 12
}
```

**Lưu ý:** File gốc nhỏ (~150K ký tự) do được tổng hợp tự động từ `drugs.json`, chỉ lấy 2 trường `question` + `answer` và không có `url`.

---

### 1.4 `cleaned_drugs.json`

| Thuộc tính | Giá trị |
|---|---|
| **Nguồn gốc** | `CrawlVinmec/vinmec_complete_data/drugs.json` |
| **Script xử lý** | `clean_data_json.py` |
| **Số bản ghi gốc** | 192 bản ghi thuốc |
| **Số bản ghi sau xử lý** | 192 bản ghi |

**Mục đích:** Chứa thông tin chi tiết từng loại thuốc theo chuẩn dược học: dạng bào chế, nhóm thuốc, chỉ định, chống chỉ định, tác dụng phụ, liều dùng, lưu ý sử dụng. Bổ sung kiến thức dược học cho model.

**Dữ liệu gốc có các trường:** `url`, `name`, `formulation`, `drug_group`, `indication`, `contraindication`, `precaution`, `side_effects`, `dosage`, `usage_notes`, `references`, `related_topics`, `full_text`

> ⚠️ **Trường `full_text` bị bỏ qua** khi xử lý vì nó là bản tổng hợp của tất cả các trường khác — nếu giữ lại sẽ gây double content khi embedding.

**Cấu trúc mỗi bản ghi sau xử lý:**
```json
{
  "id": 1,
  "url": "https://www.vinmec.com/vie/thuoc/...",
  "original_text": "Abobotulinum Toxin A Bột pha tiêm: Dysport 300 UI. ...",
  "cleaned_text": "văn bản đã làm sạch (từng trường, bỏ full_text)",
  "word_count": 450
}
```

---

### 1.5 `cleaned_hellobacsi_articles.json`

| Thuộc tính | Giá trị |
|---|---|
| **Nguồn gốc** | `CrawlHelloBacSi/hellobacsi_data_{1,2,3,4,5,7,8}/articles_*.json` (7 thư mục) |
| **Script xử lý** | `preprocess_hellobacsi.py` |
| **Số bản ghi gốc (trước dedup)** | 8,889 bản ghi |
| **Số bản ghi sau dedup** | 8,679 bản ghi (loại 210 trùng) |
| **Tổng số từ** | 8,988,819 từ |
| **Từ duy nhất** | 113,001 từ |
| **TB từ/bản ghi** | 715 từ |

**Mục đích:** Chứa bài viết y tế từ HelloBacSi — đây là file lớn nhất và đa dạng nhất về chủ đề. Bao gồm nhiều danh mục sức khỏe (vắc-xin, tim mạch, tiêu hóa, sức khỏe phụ nữ, nhân mục đích nuôi dạy con,...). Là nguồn dữ liệu chính cho embedding.

**Dữ liệu gốc có các trường:** `url`, `title`, `category`, `author`, `date`, `content`

**Cấu trúc mỗi bản ghi sau xử lý:**
```json
{
  "id": 1,
  "url": "https://hellobacsi.com/...",
  "category": "Vắc-xin",
  "title": "Vaccine HPV vào chương trình tiêm chủng mở rộng...",
  "original_text": "500 ký tự đầu...",
  "cleaned_text": "văn bản đã làm sạch",
  "word_count": 254
}
```

**Phân bố nguồn crawl:**

| Thư mục | Bản ghi gốc | Thêm vào (sau dedup) | Bỏ trùng |
|---|---|---|---|
| hellobacsi_data_1 | 1,958 | 1,958 | 0 |
| hellobacsi_data_2 | 264 | 264 | 0 |
| hellobacsi_data_3 | 2,028 | 2,021 | 7 |
| hellobacsi_data_4 | 1,440 | 1,271 | 169 |
| hellobacsi_data_5 | 166 | 166 | 0 |
| hellobacsi_data_7 | 2,102 | 2,087 | 15 |
| hellobacsi_data_8 | 931 | 912 | 19 |
| **Tổng** | **8,889** | **8,679** | **210** |

---

## 2. File hỗ trợ / trung gian

### 2.1 `hellobacsi_merged_raw.json`

| Thuộc tính | Giá trị |
|---|---|
| **Tạo bởi** | `preprocess_hellobacsi.py` → `merge_and_deduplicate_hellobacsi()` |
| **Nội dung** | 8,679 bản ghi raw (chưa làm sạch) từ tất cả 7 thư mục, đã dedup |

**Mục đích:** File merged vật lý để **tái sử dụng** — không cần đọc lại 7 thư mục mỗi lần chạy lại pipeline. Giữ nguyên format gốc (chưa tokenize, chưa remove stopwords).

---

## 3. File thống kê

| File | Nội dung | Tạo bởi |
|---|---|---|
| `raw_data_statistics.json` | Thống kê dữ liệu Vinmec **trước** làm sạch (số bản ghi, ký tự, trường dữ liệu) | `clean_data_json.py` |
| `hellobacsi_raw_stats.json` | Thống kê HelloBacSi **trước** làm sạch, theo từng thư mục, kèm số bản ghi trùng đã loại | `preprocess_hellobacsi.py` |
| `cleaned_diseases_stats.json` | Thống kê `cleaned_diseases.json`: tổng từ, từ duy nhất, top 50 từ, phân phối tần suất | `clean_data_json.py` |
| `cleaned_articles_stats.json` | Thống kê `cleaned_articles.json` | `clean_data_json.py` |
| `cleaned_drug_qa_pairs_stats.json` | Thống kê `cleaned_drug_qa_pairs.json` | `clean_data_json.py` |
| `cleaned_drugs_stats.json` | Thống kê `cleaned_drugs.json` | `clean_data_json.py` |
| `cleaned_hellobacsi_stats.json` | Thống kê `cleaned_hellobacsi_articles.json` kèm phân bố theo danh mục, top 50 từ | `preprocess_hellobacsi.py` |
| `potential_typos.json` | Danh sách từ nghi ngờ sai chính tả (xuất hiện < 3 lần, ngắn hoặc có số) | `clean_data_json.py` |

---

## 4. Tổng hợp toàn bộ corpus

| Nguồn | File cleaned | Bản ghi | Tổng từ |
|---|---|---|---|
| Vinmec - Bệnh | `cleaned_diseases.json` | 386 | 1,470,882 |
| Vinmec - Bài viết | `cleaned_articles.json` | 2,215 | 3,477,588 |
| Vinmec - Hỏi đáp thuốc | `cleaned_drug_qa_pairs.json` | 546 | 24,674 |
| Vinmec - Thuốc | `cleaned_drugs.json` | 192 | *(chưa có stats)* |
| HelloBacSi | `cleaned_hellobacsi_articles.json` | 8,679 | 8,988,819 |
| **TỔNG** | | **12,018** | **~13,961,963+** |

---

## 5. Quy trình làm sạch (áp dụng cho tất cả file)

```
Văn bản thô
    ↓ Lowercase
    ↓ Chuẩn hóa thuật ngữ y tế (bác sĩ → bác_sĩ, bệnh nhân → bệnh_nhân, ...)
    ↓ Loại bỏ dấu câu  ! ( ) - [ ] { } ; : ' " \ , < > . / ? @ # $ ^ & * _ ~
    ↓ Word tokenize (underthesea — tách từ tiếng Việt)
    ↓ Loại bỏ stop words (vietnamese-stopwords.txt — ~2,000 từ)
    ↓ Loại bỏ từ rỗng và từ có độ dài ≤ 1 ký tự
    ↓
cleaned_text (lưu vào các file cleaned_*.json)
```

**Dedup (chỉ HelloBacSi):** Loại bỏ bản ghi trùng lặp theo `url` trước khi làm sạch.

**skip_fields (chỉ drugs.json):** Bỏ qua trường `full_text` để tránh double content.
