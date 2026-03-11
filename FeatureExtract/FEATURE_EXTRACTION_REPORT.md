# Báo cáo Trích xuất Đặc trưng Văn bản (Feature Extraction)

**Dữ liệu:** `merged_cleaned_data.json` — 12,018 bản ghi văn bản y tế tiếng Việt  
**Nguồn:** Vinmec (bệnh lý, bài viết, hỏi đáp thuốc, thuốc) + HelloBacSi  
**Tổng từ:** 14,007,148 từ | Trung bình: 1,166 từ/bản ghi | Median: 992 từ | Min: 2 | Max: 120,090

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
| Sparsity | 34.5% |
| Top từ (số doc chứa) | `thể` (500), `điềutrị` (498), `có` (497), `bệnh` (496), `cơ` (492) |
| Không phân biệt tần số | Từ xuất hiện 1 lần = từ xuất hiện 100 lần |

**Hình ảnh (`onehot_result.png`):**

*Biểu đồ trái — Heatmap 20 tài liệu × 30 từ đầu:*
- Nền màu **xanh đậm** chiếm phần lớn diện tích — cho thấy hầu hết các từ trong top vocab đều **xuất hiện** trong gần như mọi tài liệu (ô =1). Chỉ một vài ô trắng lẻ tẻ (ô =0) mới cho thấy từ vắng mặt.
- Điều này phản ánh sparsity thực tế chỉ **34.5%** (thấp hơn nhiều so với kỳ vọng) — do vocab được chọn là **top 200 từ PHỔ BIẾN NHẤT**, tất nhiên chúng xuất hiện rộng rãi.
- Các cột bên phải (từ ít phổ biến hơn trong top 30) mới bắt đầu xuất hiện vài ô trắng.

*Biểu đồ phải — Bar chart top 20 từ theo presence count:*
- `thể` đứng đầu với **500/500** tài liệu (100%), tiếp theo `điềutrị` (~498), `có` (~497), `bệnh` (~496), `cơ` (~492).
- Toàn bộ 20 từ đều xuất hiện trong **>450 tài liệu** (>90%) — khẳng định đây là vocab lõi của corpus y tế.
- Các từ cuối danh sách như `trường`, `định`, `hiểu` vẫn có mặt trong ~450 doc.

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
| Sparsity | 93.6% |
| Từ phổ biến nhất | `thể` (226,017), `thuốc` (195,688), `có` (165,268), `dụng` (141,852), `bệnh` (139,724) |
| Trung bình từ/doc | 1,124 | Max: 112,725 |

**Hình ảnh (`count_vectorizer_result.png`):**

*Biểu đồ trái — Bar chart top 30 từ theo tổng tần số:*
- `thể` dẫn đầu với **226,017** lần xuất hiện — đây là từ ghép tiếng Việt thường đứng sau động từ (`có thể`, `cơ thể`), không mang nhiều ý nghĩa độc lập.
- `thuốc` xếp thứ 2 (**195,688**) phản ánh trực tiếp đặc trưng corpus y tế — dữ liệu chứa nhiều bài thuốc, hướng dẫn dùng thuốc.
- Từ vị trí 3 trở đi (`có`, `dụng`, `bệnh`...) tần số giảm dần nhưng vẫn rất cao (60k–165k).
- Đây chủ yếu là **stop words y tế** hoặc từ hư — không phân biệt được tầm quan trọng thực sự của tài liệu.

*Biểu đồ phải — Histogram phân phối tổng từ/tài liệu:*
- Phân phối **cực kỳ lệch phải** (right-skewed): cột đầu tiên (~0–2,000 từ) chứa tới **>10,000 tài liệu**, chiếm áp đảo.
- Đường đỏ `Mean=1124` nằm xa về phía phải so với đỉnh phân phối — bị kéo bởi một số bản ghi rất dài.
- Phần đuôi dài từ 10,000 → 100,000+ từ: đây là các bài bệnh lý chuyên sâu từ Vinmec có nội dung rất dài, max lên đến **112,725** từ/tài liệu.

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
| n | Top n-gram phổ biến (tần số) | Ghi chú |
|---|---|---|
| Unigram | `thể` (226,017), `thuốc` (195,688), `có` (165,268) | Từ y tế chung |
| Bigram | `có thể` (158,215), `sử dụng` (78,095), `tình trạng` (39,255) | Cụm từ có nghĩa |
| Trigram | `sử dụng thuốc` (21,756), `có thể gây` (15,350), `tác dụng phụ` (13,953) | Cụm ngữ cảnh cụ thể |

**Hình ảnh (`ngrams_result.png`):**

*Biểu đồ 1 — Top 10 Unigram (màu xanh dương):*
- Hoàn toàn giống Count Vectorizer — `thể` (~226k) và `thuốc` (~196k) bỏ xa phần còn lại.
- Không có giá trị phân biệt ngữ nghĩa: `thể`, `có`, `dụng`, `sử` là các từ vô nghĩa khi đứng độc lập.

*Biểu đồ 2 — Top 10 Bigram (màu xanh lá):*
- `có thể` đứng đầu với **158,215** lần — vượt xa các bigram còn lại (gấp 2 lần `sử dụng`).
- `sử dụng` (78,095), `tình trạng` (39,255), `tác dụng` (37,833) là các cụm có ý nghĩa y tế rõ ràng hơn unigram.
- Bigram bắt được các cụm 2 từ có nghĩa như `người bệnh`, `nguy cơ`, `cơ thể` — phản ánh đặc trưng tiếng Việt (nhiều từ ghép 2 âm tiết).

*Biểu đồ 3 — Top 10 Trigram (màu cam):*
- Tần số thấp hơn nhiều (max ~21,756 so với bigram 158,215) — trigram đặc thù hơn nên xuất hiện ít hơn.
- `sử dụng thuốc` (21,756) và `tác dụng phụ` (13,953) là các cụm thuật ngữ y học chuẩn.
- `tham khảo kiến` (6,146) xuất hiện do mẫu câu lặp lại trong template của HelloBacSi.

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
| Non-zero entries | 9,994 (99.9%) |
| Top cặp | (`có`, `thể`): 18,994 — (`sử`, `dụng`): 7,607 — (`người`, `bệnh`): 7,538 |

**Hình ảnh (`cooccurrence_result.png`):**

*Biểu đồ trái — Co-occurrence Heatmap 25×25 (log scale):*
- Trục X và Y đều là top-25 từ phổ biến: `bệnh`, `thể`, `có`, `thuốc`, `người`, `cơ`, `thường`, `điềutrị`, `dụng`...
- Các ô **đỏ đậm** (giá trị log cao nhất) tập trung rõ rệt tại giao của các cặp từ ghép tiếng Việt: `sử`↔`dụng` (hàng 8, cột 8), `cơ`↔`thể`, `nguyên`↔`nhân`, `nguy`↔`cơ` — phản ánh đúng cấu trúc từ ghép của tiếng Việt.
- Dải màu cam-vàng bao phủ rộng — do window=4 rộng, các từ phổ biến luôn xuất hiện gần nhau trong văn bản y tế.
- Phần góc trên-trái (các từ top đầu như `bệnh`, `thể`, `có`) đều có màu ấm (cam-đỏ nhạt) với nhau — chúng cùng xuất hiện trong hầu hết mọi câu.

*Biểu đồ phải — Bar chart Top 20 từ theo tổng co-occurrence:*
- `bệnh` đứng đầu với tổng co-occurrence **>100,000** — từ trung tâm của toàn bộ corpus y tế, cùng xuất hiện với gần như mọi từ khác.
- `thể` xếp thứ 2 (~97,000), `có` thứ 3 (~72,000).
- Từ vị trí 4 trở đi giảm dần: `thuốc` (~57k), `cơ` (~50k), `người` (~46k), `điềutrị` (~43k).
- Biểu đồ này phản ánh **mức độ trung tâm** của mỗi từ trong mạng ngữ nghĩa y tế, không hẳn là tần số xuất hiện thuần túy.

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
| Non-zero / tài liệu | Trung bình 342 bucket (min: 2, max: 2,808) |
| Sparsity | 99.478% |
| Hash collision | Có thể xảy ra, tỉ lệ thấp với N = 65,536 |

**Hình ảnh (`hash_vectorizer_result.png`):**

*Biểu đồ trái — Histogram phân phối số bucket active/tài liệu:*
- Phân phối có dạng **chuông lệch phải nhẹ**, tập trung mạnh ở khoảng **300–500 bucket**.
- Đỉnh phân phối ở ~350 bucket, đường đỏ `Mean=342` trùng gần với đỉnh — phân phối tương đối đều đặn.
- Đuôi phải kéo dài đến ~2,500 bucket — tương ứng với các bài viết dài hàng chục nghìn từ.
- Một nhóm nhỏ tài liệu (~800 doc) có <200 bucket: đây là các bản ghi ngắn (hỏi đáp thuốc, tên thuốc đơn giản).

*Biểu đồ phải — Cosine Similarity heatmap 50 tài liệu đầu:*
- Nền màu **xanh rất nhạt** (similarity ~0) chiếm đa số — phản ánh sự thưa thớt cực cao (99.478%).
- **Đường chéo chính** màu xanh đậm (similarity=1) — tài liệu tự so với chính nó.
- Xuất hiện **vài cụm nhỏ màu trung bình** (similarity 0.4–0.8) ở một số cặp tài liệu nhất định — these là các bài viết cùng chủ đề bệnh hoặc cùng template.
- Tổng thể hash vector khó phân biệt tài liệu tốt (do collision) — similarity map ít cấu trúc hơn TF-IDF.

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
| Sparsity | 96.8% |
| IDF thấp nhất (phổ biến) | `thể` (1.0566), `có` (1.0593), `thường` (1.1176), `người` (1.1305) |
| IDF cao nhất (đặc trưng) | `quetiapin` (9.2956), `hydrea` (9.2956), `cefepim` (9.2956) — tên thuốc cụ thể |

**Hình ảnh (`tfidf_result.png`):**

*Biểu đồ trái — Histogram phân phối giá trị IDF:*
- IDF dao động từ **~1.05** (từ cực phổ biến) đến **~9.30** (từ cực hiếm), range rộng phản ánh sự đa dạng của vocab.
- Phân phối có dạng **lệch phải** với đỉnh tập trung ở dải **7–8**: đây là đa số từ xuất hiện rải rác ở ít tài liệu.
- Phần đuôi thấp ở IDF ~1–3: chỉ vài trăm từ cực phổ biến (stop words y tế). Đây chính là các từ bị TF-IDF tự động giảm trọng số.
- Mức IDF ~9.3 là giới hạn tối đa — ứng với từ chỉ xuất hiện đúng `min_df=2` tài liệu trong 12,018.

*Biểu đồ giữa — Bar chart Top 20 từ IDF cao nhất:*
- Tất cả 20 từ đều có IDF = **9.2956** (bằng nhau) — đây là giá trị IDF tối đa khi từ chỉ xuất hiện đúng 2 tài liệu.
- Toàn bộ là **tên thuốc đặc thù**: `quetiapin`, `hydrea`, `cefepim`, `cefradine`, `pletaal`, `enzalutamide`... — các thuốc hiếm/chuyên biệt chỉ được đề cập trong 1–2 bài viết.
- Điều này cho thấy TF-IDF hoạt động đúng: chính xác phân biệt được từ đặc trưng cho từng tài liệu cụ thể.

*Biểu đồ phải — Cosine Similarity 100 tài liệu đầu (TF-IDF):*
- Ma trận màu **cam-đỏ đậm** tổng thể — TF-IDF similarity cao hơn Hash rõ rệt, phản ánh corpus y tế có nội dung đồng nhất.
- **Đường chéo đen** (tự so với chính nó, đã zero) nổi bật trên nền cam.
- Xuất hiện **các vùng sáng hơn** (similarity 0.4–0.6) dải theo hàng/cột — nhóm tài liệu cùng chủ đề (bệnh tim mạch, thuốc kháng sinh...) cluster lại tự nhiên.
- So với Hash Vectorizer, ma trận TF-IDF có **cấu trúc rõ hơn nhiều** — minh chứng cho hiệu quả vượt trội của trọng số IDF trong phân biệt nội dung.

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
| Vocab size | 35,709 từ |
| Vector size | 100 chiều |
| Từ gần `bệnh` | `cácbệnh` (0.777), `đượcbệnh` (0.719), `nhiềubệnh` (0.709) |
| Từ gần `thuốc` | `loạithuốc` (0.684), `toa` (0.646) |
| Từ gần `điềutrị` | `chẩnđoán` (0.727), `chữa` (0.671) |

**Hình ảnh:**

*`word2vec_tsne.png`* — t-SNE 2D (300 từ):
- Scatter plot 300 từ được giảm chiều từ 100D xuống 2D bằng t-SNE (perplexity=30, 500 iter).
- Không gian **không có cluster hoàn toàn rõ ràng** do corpus y tế đồng nhất về chủ đề — phần lớn từ đặc thù cho lĩnh vực này.
- Quan sát được **vài cụm cục bộ**: `máu` và `huyết` nằm gần nhau (góc trên-trái), `điềutrị` và `bácsĩ` gần nhau (góc trái-giữa), `viêm` và `nhiễm` gần nhau (phía dưới-trái).
- Từ như `thuốc`, `dụng`, `dùng` co cụm ở vùng trung tâm — phản ánh chúng xuất hiện trong nhiều ngữ cảnh khác nhau.
- Một số điểm nằm biệt lập (outlier) ở rìa đồ thị là các từ ít xuất hiện hoặc từ đặc thù của một nhóm tài liệu nhỏ.

*`word2vec_similarity.png`* — Cosine Similarity heatmap 14 × 14:
- **Nhóm bệnh lý – điều trị** có similarity cao: `bệnh`↔`triệuchứng` (**0.65**), `bệnh`↔`điềutrị` (**0.61**), `điềutrị`↔`chẩnđoán` (**0.73** — cao nhất ngoài self-similarity).
- **Nhóm triệu chứng**: `viêm`↔`đau` (**0.62**), `đau`↔`sốt` (**0.56**) — phản ánh các triệu chứng thường đi kèm nhau trong văn bản lâm sàng.
- **Nhóm bộ phận cơ thể**: `tim`↔`gan` (**0.45**), `tim`↔`huyết` (**0.51**) — cùng thuộc nhóm nội tạng/tuần hoàn.
- **Thấp nhất**: `bácsĩ`↔`gan` (**0.09**), `xétnghiệm`↔`đau` (**0.14**) — các từ thuộc vai trò/hành động khác biệt hẳn với triệu chứng.
- `điềutrị`↔`chẩnđoán` similarity **0.73**: model học được mối liên hệ chặt giữa hai thao tác y khoa này — không thể có được từ Count Vectorizer hay TF-IDF.

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
