# Báo cáo Trích xuất Đặc trưng Văn bản (Feature Extraction)

**Dữ liệu:** `merged_cleaned_data.json` — 12,018 bản ghi văn bản y tế tiếng Việt  
**Nguồn:** Vinmec (bệnh lý, bài viết, hỏi đáp thuốc, thuốc) + HelloBacSi  
**Tổng từ:** ~9,744,309 từ | Trung bình: 810 từ/bản ghi

---

## 1. One-hot Encoding

### Nguyên lý
Mỗi từ trong từ điển kích thước $|V|$ được ánh xạ thành một vector nhị phân kích thước $|V|$, trong đó chỉ vị trí tương ứng với từ đó bằng `1`, tất cả còn lại bằng `0`.

$$\text{one-hot}(\text{bệnh}) = [0, 0, \ldots, 1, \ldots, 0]$$

Đối với tài liệu: vector là **hợp (union)** của các one-hot vector của từng từ — tức là `1` nếu từ *xuất hiện ít nhất 1 lần*, `0` nếu không.

### Cài đặt
- Vocab: top **200 từ** phổ biến nhất (từ 500 tài liệu mẫu)
- Shape output: `(500, 200)`
- Tool: `sklearn.preprocessing.MultiLabelBinarizer`

### Kết quả & Nhận xét
| Chỉ số | Giá trị |
|---|---|
| Shape | 500 × 200 |
| Sparsity | ~85–90% (phần lớn là 0) |
| Không phân biệt tần số | Từ xuất hiện 1 lần = từ xuất hiện 100 lần |

**Hình ảnh (`onehot_result.png`):**
- *Heatmap* 20 tài liệu × 30 từ: cho thấy rõ sự thưa thớt — phần lớn ô màu trắng (không xuất hiện)
- *Bar chart* từ xuất hiện nhiều nhất: các từ như `bệnh`, `bệnhnhân`, `điềutrị` chiếm ưu thế tuyệt đối

**Ưu điểm:**  
- Đơn giản, dễ hiểu, không cần training  
- Tốt cho bài toán phân loại nhị phân với vocab nhỏ

**Nhược điểm:**  
- Không phân biệt tần số; không nắm bắt ngữ nghĩa  
- Bùng nổ chiều khi vocab lớn (curse of dimensionality)  
- Không phù hợp với corpus tiếng Việt y tế đa dạng

---

## 2. Count Vectorizing (Bag-of-Words)

### Nguyên lý
Đếm số lần xuất hiện của từng từ trong mỗi tài liệu. Kết quả là ma trận $D \times V$ trong đó $D$ là số tài liệu, $V$ là kích thước vocab:

$$\text{count}(d, w) = \text{số lần } w \text{ xuất hiện trong } d$$

### Cài đặt
- `max_features=5000`, `min_df=2` (từ xuất hiện ít nhất 2 tài liệu)
- Toàn bộ 12,018 tài liệu
- Tool: `sklearn.feature_extraction.text.CountVectorizer`

### Kết quả & Nhận xét
| Chỉ số | Giá trị |
|---|---|
| Shape | 12,018 × 5,000 |
| Sparsity | ~97–99% |
| Từ phổ biến nhất | `bệnh`, `bệnhnhân`, `điềutrị`, `triệuchứng` |

**Hình ảnh (`count_vectorizer_result.png`):**
- *Bar chart* top-30 từ: khẳng định corpus y tế — các từ đặc trưng xuất hiện hàng trăm nghìn lần
- *Histogram* tổng từ/tài liệu: phân phối lệch phải, phần lớn tài liệu có 200–1000 từ, một số bản ghi từ bệnh lý Vinmec có đến 70k+ từ

**Ưu điểm:**  
- Giữ thông tin tần số (hơn One-hot)  
- Đơn giản, nhanh, tốt cho baseline

**Nhược điểm:**  
- Từ phổ biến khắp nơi (*stop words* như `bệnh`, `của`) được trọng số cao giả tạo  
- Không nắm bắt thứ tự, ngữ cảnh, ngữ nghĩa

---

## 3. N-grams

### Nguyên lý
Mở rộng Bag-of-Words sang chuỗi n từ liên tiếp:
- **Unigram (n=1):** `bệnh`, `tim`, `mạch`
- **Bigram (n=2):** `bệnh tim`, `tim mạch`
- **Trigram (n=3):** `bệnh tim mạch`

N-gram nắm bắt ngữ cảnh cục bộ tốt hơn unigram.

### Cài đặt
- `max_features=5000`, `min_df=2` cho mỗi n
- Toàn bộ 12,018 tài liệu
- Tool: `CountVectorizer(ngram_range=(n, n))`

### Kết quả & Nhận xét
| n | Top bigram phổ biến | Ghi chú |
|---|---|---|
| Unigram | `bệnh`, `bệnhnhân`, `điềutrị` | Từ y tế chung |
| Bigram | `bệnh lý`, `triệu chứng`, `bệnh nhân` | Cụm từ có nghĩa |
| Trigram | `chẩn đoán điều trị`, `bệnh nhân cần` | Cụm ngữ cảnh cụ thể |

**Hình ảnh (`ngrams_result.png`):**
- 3 biểu đồ ngang so sánh top-10 n-gram của từng loại
- Bigram và trigram bắt được các cụm thuật ngữ y học có nghĩa nhiều hơn unigram đơn lẻ

**Ưu điểm:**  
- Bắt được ngữ cảnh cục bộ, phân biệt được `không đau` vs `đau`  
- Bigram và trigram phù hợp với tiếng Việt có nhiều từ ghép

**Nhược điểm:**  
- Kích thước vocab tăng đột biến (n-gram tạo ra rất nhiều combination)  
- Vẫn không nắm được quan hệ ngữ nghĩa xa

---

## 4. Co-occurrence Matrix

### Nguyên lý
Đếm số lần từ $w_i$ và $w_j$ **cùng xuất hiện trong một cửa sổ** (window) kích thước $k$ xung quanh nhau:

$$C[i][j] = \text{số lần } w_i \text{ xuất hiện trong khoảng cách } k \text{ với } w_j$$

Ma trận vuông kích thước $|V| \times |V|$. Nền tảng lý thuyết của GloVe.

### Cài đặt
- Vocab: top **100 từ** (từ 500 tài liệu mẫu)
- Window size: **4** (4 từ trái và phải)
- Shape output: `(100, 100)`

### Kết quả & Nhận xét
| Chỉ số | Giá trị |
|---|---|
| Shape | 100 × 100 |
| Non-zero entries | >80% (với top từ phổ biến) |
| Top cặp | (`bệnh`, `bệnhnhân`), (`điềutrị`, `bệnhnhân`) |

**Hình ảnh (`cooccurrence_result.png`):**
- *Heatmap log-scale* 25×25: vùng đậm tập trung ở góc trái-trên (các từ phổ biến nhất luôn cùng xuất hiện)
- *Bar chart* tổng co-occurrence: `bệnhnhân` và `bệnh` có tổng đồng xuất hiện cao nhất

**Ưu điểm:**  
- Phản ánh quan hệ ngữ nghĩa dựa trên ngữ cảnh  
- Nền tảng để học word embedding kiểu GloVe

**Nhược điểm:**  
- Bộ nhớ $O(|V|^2)$ — không khả thi với vocab lớn  
- Ma trận rất thưa với vocab đầy đủ

---

## 5. Hash Vectorizing

### Nguyên lý
Ánh xạ mỗi từ vào một **bucket** cố định bằng hàm băm, **không cần lưu từ điển**:

$$\text{index}(w) = \text{hash}(w) \mod N$$

trong đó $N$ là số bucket (chiều vector). Giá trị trong mỗi bucket là tổng tần số (hoặc có dấu xen kẽ để tránh collision) của tất cả từ băm vào đó.

### Cài đặt
- `n_features=2^16 = 65,536` buckets  
- `norm='l2'` (chuẩn hoá L2)
- Tool: `sklearn.feature_extraction.text.HashingVectorizer`

### Kết quả & Nhận xét
| Chỉ số | Giá trị |
|---|---|
| Shape | 12,018 × 65,536 |
| Non-zero / tài liệu | ~500–1000 bucket |
| Sparsity | >99.9% |
| Hash collision | Có thể xảy ra, tỉ lệ thấp với N lớn |

**Hình ảnh (`hash_vectorizer_result.png`):**
- *Histogram* số bucket active: phân phối khá đối xứng xung quanh giá trị trung bình
- *Cosine similarity heatmap* 50 tài liệu: phần lớn gần 0, một vài cụm tài liệu có similarity cao (cùng chủ đề bệnh)

**Ưu điểm:**  
- **Không cần từ điển** — xử lý streaming, dữ liệu mới không cần retrain  
- Bộ nhớ cố định, tốc độ nhanh  
- Tự xử lý từ mới (out-of-vocabulary)

**Nhược điểm:**  
- Không thể giải mã ngược (bucket → từ)  
- Hash collision: hai từ khác nhau có thể vào cùng bucket  
- Không thể tính IDF

---

## 6. TF-IDF

### Nguyên lý
Kết hợp hai đại lượng:

$$\text{TF}(t,d) = \log(1 + \text{count}(t,d))$$

$$\text{IDF}(t) = \log\frac{N + 1}{df(t) + 1} + 1$$

$$\text{TF-IDF}(t,d) = \text{TF}(t,d) \times \text{IDF}(t)$$

Từ xuất hiện nhiều trong 1 tài liệu nhưng hiếm trên toàn corpus → trọng số cao.  
Từ xuất hiện ở mọi tài liệu (như `bệnh`) → IDF thấp → trọng số thấp.

### Cài đặt
- `max_features=10,000`, `min_df=2`, `sublinear_tf=True`
- Toàn bộ 12,018 tài liệu
- Tool: `sklearn.feature_extraction.text.TfidfVectorizer`

### Kết quả & Nhận xét
| Chỉ số | Giá trị |
|---|---|
| Shape | 12,018 × 10,000 |
| IDF thấp nhất (phổ biến) | `bệnh`, `bệnhnhân`, `điềutrị` |
| IDF cao nhất (đặc trưng) | Tên thuốc cụ thể, thuật ngữ hiếm |

**Hình ảnh (`tfidf_result.png`):**
- *Histogram IDF*: phân phối lệch phải — nhiều từ có IDF trung bình, ít từ cực hiếm
- *Bar chart IDF cao*: các tên thuốc đặc thù, thuật ngữ chuyên ngành hẹp xuất hiện
- *Cosine similarity 100 doc*: rõ hơn Hash Vectorizer — các cụm bệnh lý tương tự cluster lại

**Ưu điểm:**  
- **Phương pháp hiệu quả nhất** trong nhóm statistical cho retrieval  
- Giảm nhiễu từ stop words tự động qua IDF  
- Tiêu chuẩn de facto trong Information Retrieval và text classification

**Nhược điểm:**  
- Vẫn không nắm bắt ngữ nghĩa (synonym, polysemy)  
- Vector thưa, chiều cao

---

## 7. Word Embedding (Word2Vec)

### Nguyên lý
Học **vector dày đặc** (dense) kích thước cố định cho từng từ bằng mạng nơ-ron nông (shallow neural network). Hai kiến trúc:

- **Skip-gram** (dùng ở đây): dự đoán các từ ngữ cảnh từ từ trung tâm  
- **CBOW:** dự đoán từ trung tâm từ ngữ cảnh

$$\text{maximize} \sum_{w \in W} \sum_{-k \leq j \leq k, j \neq 0} \log P(w_{t+j} | w_t)$$

Từ có nghĩa tương tự → vector gần nhau (cosine similarity cao).

### Cài đặt
- `vector_size=100`, `window=5`, `min_count=3`
- `sg=1` (Skip-gram), `epochs=10`
- Toàn bộ 12,018 tài liệu
- Tool: `gensim.models.Word2Vec`

### Kết quả & Nhận xét
| Chỉ số | Giá trị |
|---|---|
| Vocab size | ~40,000–60,000 từ |
| Vector size | 100 chiều |
| Từ gần `bệnh` | `bệnh lý`, `mắc bệnh`, `bệnhnhân` |
| Từ gần `thuốc` | tên thuốc cụ thể, `liều dùng` |

**Hình ảnh:**

*`word2vec_tsne.png`* — t-SNE 2D:
- Các từ cùng chủ đề (thuốc, triệu chứng, bộ phận cơ thể) tự nhiên cluster lại theo nhóm
- Không gian embedding phản ánh cấu trúc ngữ nghĩa của corpus y tế

*`word2vec_similarity.png`* — Similarity heatmap:
- `bệnh` ↔ `bệnhnhân`: similarity cao (~0.7–0.8)
- `điềutrị` ↔ `thuốc`: similarity cao
- `gan` ↔ `tim`: similarity trung bình (cùng nhóm bộ phận cơ thể)

**Ưu điểm:**  
- **Nắm bắt ngữ nghĩa** — từ đồng nghĩa có vector gần nhau  
- Vector dày đặc, chiều thấp (100) → hiệu quả tính toán cho downstream tasks  
- Phù hợp làm initialization cho các mô hình deep learning

**Nhược điểm:**  
- Mỗi từ chỉ có 1 vector (không xử lý đa nghĩa — polysemy)  
- Cần lượng dữ liệu đủ lớn để học tốt  
- Không xử lý từ ngoài vocab (OOV) khi inference

---

## Tổng kết so sánh

| Phương pháp | Chiều vector | Sparse/Dense | Giữ thứ tự | Nắm ngữ nghĩa | Phù hợp cho |
|---|---|---|---|---|---|
| One-hot | \|V\| = 200 | Sparse | Không | Không | Baseline đơn giản |
| Count Vectorizer | 5,000 | Sparse | Không | Không | Phân loại văn bản |
| N-grams (bigram) | 5,000 | Sparse | Cục bộ | Một phần | Text classification |
| Co-occurrence | V×V = 100×100 | Dense (nhỏ) | Không | Có | Phân tích quan hệ từ |
| Hash Vectorizer | 65,536 | Sparse | Không | Không | Streaming / dữ liệu lớn |
| TF-IDF | 10,000 | Sparse | Không | Một phần | **Search / Retrieval** |
| Word2Vec | 100 | Dense | Có | **Có** | **NLP sâu, embedding** |

### Khuyến nghị cho pipeline NLP y tế này

1. **Nếu dùng cho embedding + RAG (Retrieval-Augmented Generation):**  
   → Dùng **TF-IDF** cho sparse retrieval (BM25-style) kết hợp **Word2Vec/Sentence Embedding** cho dense retrieval

2. **Nếu dùng cho text classification (phân loại bệnh):**  
   → **TF-IDF + N-grams** cho baseline; **Word2Vec** làm feature cho ML models

3. **Nếu dùng cho train language model:**  
   → `cleaned_text` trực tiếp, dùng tokenizer của model (BERT, PhoBERT, etc.)

---

## File đầu ra

| File | Mô tả |
|---|---|
| `feature_extraction.ipynb` | Notebook code đầy đủ 7 phương pháp |
| `onehot_result.png` | Heatmap + word presence One-hot |
| `count_vectorizer_result.png` | Top từ + phân phối độ dài |
| `ngrams_result.png` | Top 10 uni/bi/trigram |
| `cooccurrence_result.png` | Heatmap co-occurrence + top từ |
| `hash_vectorizer_result.png` | Phân phối bucket + similarity |
| `tfidf_result.png` | Phân phối IDF + top từ + similarity |
| `word2vec_tsne.png` | t-SNE 2D của word embedding |
| `word2vec_similarity.png` | Heatmap similarity các từ y tế |
