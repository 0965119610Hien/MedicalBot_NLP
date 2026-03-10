# BÁO CÁO THU THẬP VÀ TIỀN XỬ LÝ DỮ LIỆU
## Xử lý Ngôn ngữ Tự nhiên (NLP) - Dữ liệu Y tế Tiếng Việt

---

# SLIDE 1: GIỚI THIỆU TỔNG QUAN

## Mục tiêu dự án
- Thu thập dữ liệu y tế tiếng Việt từ các nguồn uy tín
- Tiền xử lý dữ liệu phục vụ các bài toán NLP (chatbot y tế, hỏi đáp tự động, ...)
- Xây dựng bộ dữ liệu huấn luyện (training data) chất lượng

## Hai nguồn dữ liệu
| Nguồn | Website | Mô tả |
|-------|---------|-------|
| **HelloBacSi** | hellobacsi.com | Cổng thông tin sức khỏe hàng đầu Việt Nam |
| **Vinmec** | vinmec.com | Hệ thống bệnh viện & thông tin y khoa Vinmec |

---

# SLIDE 2: NGUỒN DỮ LIỆU THU THẬP

## 1. HelloBacSi (hellobacsi.com)
- **Công cụ crawl:** Python + Selenium (headless Chrome) + BeautifulSoup
- **Nội dung:** Bài viết sức khỏe dạng long-form, hướng dẫn y tế, thông tin bệnh lý
- **8 script crawl, 7 thư mục dữ liệu** (data_1 → data_8, không có data_6)

| STT | Thư mục | Số bài viết | Chủ đề chính |
|-----|---------|-------------|-------------|
| 1 | hellobacsi_data_1 | 1,958 | Đa danh mục (Vắc-xin, Tim mạch, Tiêu hóa, Truyền nhiễm...) |
| 2 | hellobacsi_data_2 | 264 | Bệnh tim mạch |
| 3 | hellobacsi_data_3 | 2,028 | Đa danh mục (Tâm lý, Dị ứng, Tai mũi họng, Tiêu hóa...) |
| 4 | hellobacsi_data_4 | 1,440 | Đa danh mục (Dị ứng, Cơ xương khớp, Máu, Truyền nhiễm...) |
| 5 | hellobacsi_data_5 | 166 | Bệnh thận & Đường tiết niệu |
| 6 | hellobacsi_data_7 | 2,102 | Đa danh mục (Tai mũi họng, Dược liệu, Da liễu, Phụ nữ...) |
| 7 | hellobacsi_data_8 | 931 | Đa danh mục (Thói quen lành mạnh, Ung thư, Tiểu đường...) |
| | **TỔNG** | **8,889** | |

## 2. Vinmec (vinmec.com)
- **Công cụ crawl:** Python + Requests + BeautifulSoup + Selenium
- **3 loại dữ liệu:**

| STT | Loại dữ liệu | Số bản ghi | Tổng ký tự | TB ký tự/bản ghi |
|-----|--------------|-----------|------------|-------------------|
| 1 | Bài viết y khoa | 2,215 | 22,621,332 | 10,213 |
| 2 | Bệnh lý (A-Z) | 386 | 8,909,785 | 23,082 |
| 3 | Cặp Hỏi-Đáp thuốc | 546 | 149,953 | 275 |
| | **TỔNG** | **3,147** | **31,681,070** | |

---

# SLIDE 3: TỔNG HỢP SỐ LIỆU THU THẬP

## Tổng quan toàn bộ dữ liệu

| Chỉ số | HelloBacSi | Vinmec | **Tổng cộng** |
|--------|-----------|--------|---------------|
| **Số bản ghi** | 8,889 | 3,147 | **12,036** |
| **Thư mục dữ liệu** | 7 folders | 1 folder | **8 folders** |
| **Script crawl** | 8 scripts | 2 scripts | **10 scripts** |
| **Loại nội dung** | Bài viết sức khỏe (20+ chủ đề) | Bài viết + Bệnh lý + Q&A thuốc | Đa dạng |

## Cấu trúc dữ liệu thu thập

**HelloBacSi** — Mỗi bài viết gồm:
- `url`, `title`, `category`, `author`, `date`, `content`

**Vinmec** — 3 dạng:
- **Bài viết:** `url`, `tieu_de`, `mo_ta`, `phan_doan[]`, `noi_dung`, `chu_de[]`
- **Bệnh lý:** `url`, `nguyen_nhan`, `yeu_to_nguy_co`, `trieu_chung`, `chan_doan`, `dieu_tri`, `phong_ngua`, `noi_dung_day_du`
- **Q&A thuốc:** `question`, `answer` (546 cặp hỏi-đáp về chỉ định & liều dùng)

---

# SLIDE 4: QUY TRÌNH TIỀN XỬ LÝ

## Pipeline xử lý 7 bước

```
Dữ liệu gốc (JSON)
    │
    ▼
[1] Gộp các trường văn bản → Tạo văn bản thống nhất
    │
    ▼
[2] Chuyển chữ thường (Lowercase)
    │
    ▼
[3] Chuẩn hóa thuật ngữ y tế (Medical Term Normalization)
    │   "bs" → "bác_sĩ", "bv" → "bệnh_viện", "bn" → "bệnh_nhân"
    │   "điều trị" → "điều_trị", "triệu chứng" → "triệu_chứng"
    │
    ▼
[4] Loại bỏ dấu câu (Remove Punctuation)
    │   Loại bỏ: ! ( ) - [ ] { } ; : ' " \ , < > . / ? @ # $ ^ & * _ ~
    │
    ▼
[5] Tách từ tiếng Việt (Vietnamese Word Tokenization)
    │   Sử dụng thư viện: underthesea.word_tokenize()
    │
    ▼
[6] Loại bỏ Stop Words (140 từ dừng tiếng Việt)
    │
    ▼
[7] Chuẩn hóa danh mục & Lưu kết quả → JSON
    │   Slug URL → Tên đầy đủ tiếng Việt
    │   'benh-tim-mach' → 'Bệnh tim mạch', 'di-ung' → 'Dị ứng'
    │   Cấu trúc: {id, url, category, original_text, cleaned_text, word_count}
```

## Các bước bổ sung
- **Phát hiện lỗi chính tả:** Xác định từ tần suất thấp, có chứa số/ký tự đặc biệt
- **Xây dựng từ điển:** Tạo bộ từ vựng từ toàn bộ dữ liệu
- **Thống kê chi tiết:** Phân tích tần suất từ, phân phối độ dài

---

# SLIDE 5: KẾT QUẢ SAU TIỀN XỬ LÝ

## Thống kê dữ liệu sau tiền xử lý

### HelloBacSi (8,889 bài viết — 7 thư mục)

| Chỉ số | 5 folder đã xử lý (4,759 bài) | Dự kiến sau khi xử lý đủ (8,889 bài) |
|--------|-------------------------------|---------------------------------------|
| Tổng số từ | 4,949,686 | **~9–10 triệu** |
| Số từ duy nhất (unique) | 69,282 | **~100,000+** |
| TB từ/bài viết | 720.6 | ~720 |
| Min / Max từ | 35 / 74,332 | — |

> ⚠️ **data_3 (2,028 bài) và data_7 (2,102 bài) đã được bổ sung vào pipeline, cần chạy lại script để có số liệu chính thức.**

**Phân bổ theo nhóm chủ đề:**
| Nhóm chủ đề | Ví dụ danh mục |
|------------|----------------|
| Bệnh lý | Tim mạch, Tiêu hóa, Hô hấp, Cơ xương khớp, Thận & Tiết niệu, Tai mũi họng, Ung thư, Truyền nhiễm, Não & TK, Máu, Dị ứng, Da liễu, Tiểu đường |
| Sức khỏe & Lối sống | Thói quen lành mạnh, Ăn uống lành mạnh, Thể dục thể thao, Lão hóa lành mạnh |
| Đặc thù | Mang thai, Phụ nữ, Nam giới, Nuôi dạy con, Tâm lý - Tâm thần |
| Chuyên ngành | Vắc-xin, Dược liệu, Thuốc |

### Vinmec (3,147 bản ghi)

| Loại dữ liệu | Tổng từ | Từ duy nhất | TB từ/bản ghi |
|--------------|---------|------------|---------------|
| Bài viết y khoa (2,215) | **3,477,588** | 45,320 | ~1,570 |
| Bệnh lý A-Z (386) | **1,470,882** | 7,108 | ~3,810 |
| Q&A thuốc (546) | **24,674** | 1,683 | ~45 |

### TỔNG HỢP

| | HelloBacSi | Vinmec | **TỔNG** |
|--|-----------|--------|----------|
| **Bản ghi** | 8,889 | 3,147 | **12,036** |
| **Tổng từ** | 9,182,068 | 4,973,144 | **~14.15 triệu từ** |
| **Từ duy nhất** | 113,003 | ~54,111 | — |

---

# SLIDE 6: TOP TỪ KHÓA PHỔ BIẾN NHẤT

## HelloBacSi — Top 10 (8,889 bài viết)

| Hạng | Từ | Số lần xuất hiện |
|------|-----|-----------------|
| 1 | thể | 160,849 |
| 2 | có | 116,031 |
| 3 | bệnh | 74,740 |
| 4 | người | 68,054 |
| 5 | dụng | 57,984 |
| 6 | cơ | 57,189 |
| 7 | thường | 54,292 |
| 8 | da | 53,258 |
| 9 | thuốc | 52,386 |
| 10 | tình | 46,099 |

## Vinmec (Bài viết) — Top 10

| Hạng | Từ | Số lần xuất hiện |
|------|-----|-----------------|
| 1 | thuốc | 134,558 |
| 2 | dụng | 77,577 |
| 3 | dùng | 52,539 |
| 4 | sử | 46,903 |
| 5 | thể | 42,418 |
| 6 | liều | 39,800 |
| 7 | bệnh | 37,802 |
| 8 | người | 35,317 |
| 9 | có | 33,911 |
| 10 | điều_trị | 32,577 |

→ **Nhận xét:** Từ khóa y tế chiếm ưu thế: "bệnh", "thuốc", "điều_trị", "triệu_chứng", "bác_sĩ"

---

# SLIDE 7: CÁCH TỔ CHỨC DỮ LIỆU

## Cấu trúc thư mục

```
NLP/
├── CrawlHelloBacSi/              ← Crawler + dữ liệu HelloBacSi
│   ├── hellobacsi_crawler_*.py   ← 8 script crawl
│   ├── hellobacsi_crawler_1.py   ← Đa danh mục
│   ├── hellobacsi_crawler_2.py   ← Tim mạch + Hô hấp + Ung thư
│   ├── hellobacsi_crawler_3.py   ← Tâm lý + Dị ứng
│   ├── hellobacsi_crawler_4.py   ← Dị ứng + Cơ xương khớp + Máu
│   ├── hellobacsi_crawler_5.py   ← Thận & Tiết niệu + Não & TK
│   ├── hellobacsi_crawler_7.py   ← Tai mũi họng
│   ├── hellobacsi_crawler_8.py   ← Thói quen lành mạnh
│   ├── hellobacsi_data_1/        ← 1,958 bài (đa danh mục)
│   │   ├── articles_1.json       ← Dữ liệu gốc
│   │   ├── training_data_1.json  ← Format huấn luyện
│   │   └── discovered_urls_1.json
│   ├── hellobacsi_data_2/        ← 264 bài (Tim mạch)
│   ├── hellobacsi_data_3/        ← 2,028 bài (Tâm lý, Dị ứng, Tiêu hóa...)
│   ├── hellobacsi_data_4/        ← 1,440 bài (Dị ứng, Cơ xương khớp...)
│   ├── hellobacsi_data_5/        ← 166 bài (Thận & Tiết niệu)
│   ├── hellobacsi_data_7/        ← 2,102 bài (Tai mũi họng, Dược liệu...)
│   └── hellobacsi_data_8/        ← 931 bài (Thói quen lành mạnh, Ung thư...)
│
├── CrawlVinmec/                  ← Crawler + dữ liệu Vinmec
│   ├── vinmec_complete_crawler.py
│   ├── vinmec_drug_crawler.py
│   └── vinmec_complete_data/
│       ├── articles.json         ← 2,215 bài viết
│       ├── diseases.json         ← 386 bệnh lý
│       ├── drugs.json            ← 192 thuốc
│       ├── drug_qa_pairs.json    ← 546 cặp Q&A
│       ├── training_data.json    ← Format huấn luyện bài viết
│       └── drug_training_data.json ← Format huấn luyện thuốc
│
└── Preprocessing/                ← Tiền xử lý
    ├── clean_data_json.py        ← Script tiền xử lý chính (Vinmec)
    ├── preprocess_hellobacsi.py  ← Script tiền xử lý HelloBacSi
    ├── preprocess.ipynb          ← Notebook thử nghiệm pipeline
    ├── vietnamese-stopwords.txt  ← 140 stop words tiếng Việt
    ├── cleaned_articles.json     ← Bài viết Vinmec đã xử lý
    ├── cleaned_diseases.json     ← Bệnh lý Vinmec đã xử lý
    ├── cleaned_drug_qa_pairs.json← Q&A thuốc đã xử lý
    ├── cleaned_hellobacsi_articles.json ← HelloBacSi đã xử lý
    └── *_stats.json              ← Các file thống kê
```

## Định dạng dữ liệu

### Dữ liệu gốc (Raw): JSON
```json
{
  "url": "https://hellobacsi.com/...",
  "title": "Tiêu đề bài viết",
  "category": "Vắc-xin",
  "content": "Nội dung bài viết đầy đủ..."
}
```

### Dữ liệu đã xử lý (Cleaned): JSON
```json
{
  "id": 1,
  "url": "https://...",
  "original_text": "500 ký tự đầu tiên...",
  "cleaned_text": "từ1 từ2 từ3 ... đã tách từ và loại stop words",
  "word_count": 530
}
```

### Dữ liệu huấn luyện (Training): JSON + JSONL
```json
{
  "text": "Nội dung đầy đủ của bài viết...",
  "metadata": {"url": "...", "category": "...", "title": "..."}
}
```

---

# SLIDE 8: CÔNG CỤ & THƯ VIỆN SỬ DỤNG

| Mục đích | Công cụ/Thư viện |
|----------|-----------------|
| Thu thập dữ liệu | Selenium, BeautifulSoup, Requests |
| Tách từ tiếng Việt | **underthesea** (word_tokenize) |
| Xử lý văn bản | Python built-in (re, string) |
| Lưu trữ dữ liệu | JSON, JSONL |
| Phân tích & thống kê | collections.Counter |
| Trình duyệt tự động | Chrome WebDriver (headless) |

---

# SLIDE 9: KẾT LUẬN & HƯỚNG PHÁT TRIỂN

## Kết quả đạt được
✅ Thu thập thành công **12,036 bản ghi** từ 2 nguồn y tế uy tín
✅ HelloBacSi: **8,889 bài viết**, **35 danh mục**, **9,182,068 từ**, **113,003 từ duy nhất**
✅ Vinmec: **3,147 bản ghi** gồm bài viết, bệnh lý, Q&A thuốc, ~4.97 triệu từ
✅ **Tổng cộng: ~14.15 triệu từ** sau tiền xử lý
✅ Pipeline tiền xử lý hoàn chỉnh **7 bước** — chuẩn hóa thuật ngữ y tế & danh mục tự động
✅ Dữ liệu sẵn sàng cho training (JSON + JSONL format)

## Hạn chế cần cải thiện
⚠️ Một số từ ghép y tế bị dính (e.g., "bácsĩ", "điềutrị")
⚠️ URL và timestamp còn lọt vào cleaned text ở một số bản ghi
⚠️ Nội dung trùng lặp trong một số bản ghi bệnh lý Vinmec
⚠️ Category slug trong dữ liệu gốc HelloBacSi không đồng nhất (slug vs. tên đầy đủ)

## Hướng phát triển
- Cải thiện tokenization cho thuật ngữ y tế chuyên ngành
- Loại bỏ trùng lặp nội dung (deduplication)
- Mở rộng thêm nguồn dữ liệu (thuốc, từ điển y khoa)
- Xây dựng mô hình chatbot y tế / hỏi-đáp tự động
- Fine-tune PhoBERT/ViT5 với bộ dữ liệu đã xây dựng

---

