"""
Trích xuất đặc trưng văn bản bằng Word Embedding
===================================================
Script này đọc dữ liệu đã làm sạch (merged_cleaned_data.json)
và thực hiện trích xuất đặc trưng bằng nhiều phương pháp Word Embedding:
  1. Word2Vec (Skip-gram & CBOW)
  2. TF-IDF Vectorizer
  3. Document Embedding (trung bình vector Word2Vec cho mỗi tài liệu)

Thư viện cần cài:
  pip install gensim scikit-learn matplotlib numpy
"""

import json
import os
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
from collections import Counter
from gensim.models import Word2Vec
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.manifold import TSNE

# ============================================================
# CẤU HÌNH
# ============================================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PREPROCESSING_DIR = os.path.join(BASE_DIR, '..', 'Preprocessing')
INPUT_FILE = os.path.join(PREPROCESSING_DIR, 'merged_cleaned_data.json')
OUTPUT_DIR = BASE_DIR

# Word2Vec params
W2V_VECTOR_SIZE = 200      # Số chiều embedding
W2V_WINDOW = 5             # Context window
W2V_MIN_COUNT = 3          # Bỏ qua từ xuất hiện < 3 lần
W2V_EPOCHS = 15            # Số epoch training
W2V_WORKERS = 4            # Số thread

# TF-IDF params
TFIDF_MAX_FEATURES = 10000
TFIDF_MIN_DF = 2
TFIDF_MAX_DF = 0.95


# ============================================================
# ĐỌC DỮ LIỆU
# ============================================================

def load_cleaned_data(file_path):
    """Đọc dữ liệu đã làm sạch."""
    print(f"Đọc dữ liệu từ: {os.path.basename(file_path)}")
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    print(f"  ✓ {len(data)} bản ghi")
    return data


def prepare_sentences(data):
    """Chuẩn bị danh sách câu (list of list of tokens) cho Word2Vec."""
    sentences = []
    for record in data:
        tokens = record['cleaned_text'].split()
        if tokens:
            sentences.append(tokens)
    return sentences


# ============================================================
# 1. WORD2VEC
# ============================================================

def train_word2vec(sentences, sg=1):
    """
    Train Word2Vec model.
    sg=1: Skip-gram, sg=0: CBOW
    """
    model_name = "Skip-gram" if sg == 1 else "CBOW"
    print(f"\n{'=' * 60}")
    print(f"Training Word2Vec ({model_name})")
    print(f"{'=' * 60}")
    print(f"  Vector size : {W2V_VECTOR_SIZE}")
    print(f"  Window      : {W2V_WINDOW}")
    print(f"  Min count   : {W2V_MIN_COUNT}")
    print(f"  Epochs      : {W2V_EPOCHS}")
    print(f"  Sentences   : {len(sentences)}")

    model = Word2Vec(
        sentences=sentences,
        vector_size=W2V_VECTOR_SIZE,
        window=W2V_WINDOW,
        min_count=W2V_MIN_COUNT,
        sg=sg,
        workers=W2V_WORKERS,
        epochs=W2V_EPOCHS,
    )

    vocab_size = len(model.wv)
    print(f"\n  ✓ Vocab size: {vocab_size}")
    print(f"  ✓ Vector shape: ({vocab_size}, {W2V_VECTOR_SIZE})")

    return model


def explore_word2vec(model, model_name="Word2Vec"):
    """Khám phá mô hình Word2Vec: từ tương tự, phép toán vector."""
    print(f"\n--- Khám phá {model_name} ---")

    # Từ y tế để kiểm tra
    medical_words = ['bệnh', 'thuốc', 'triệu_chứng', 'bác_sĩ', 'xét_nghiệm',
                     'viêm', 'đau', 'sốt', 'huyết', 'tim', 'gan', 'phổi',
                     'ung_thư', 'nhiễm_trùng', 'phẫu_thuật', 'điều_trị']

    print(f"\n  Top 10 từ tương tự:")
    for word in medical_words:
        if word in model.wv:
            similar = model.wv.most_similar(word, topn=5)
            similar_str = ', '.join([f"{w} ({s:.2f})" for w, s in similar])
            print(f"    {word:<20s} → {similar_str}")

    # Phép toán vector (nếu từ tồn tại)
    print(f"\n  Phép toán vector:")
    analogies = [
        ('bác_sĩ', 'bệnh_viện', 'giáo_viên'),
        ('thuốc', 'bệnh', 'vắc_xin'),
        ('tim', 'tim_mạch', 'gan'),
    ]
    for pos1, pos2, neg in analogies:
        if all(w in model.wv for w in [pos1, pos2, neg]):
            result = model.wv.most_similar(positive=[pos2, neg], negative=[pos1], topn=3)
            result_str = ', '.join([f"{w} ({s:.2f})" for w, s in result])
            print(f"    {pos2} - {pos1} + {neg} ≈ {result_str}")


def plot_word2vec_similarity(model, output_path):
    """Vẽ heatmap cosine similarity giữa các từ y tế."""
    medical_words = ['bệnh', 'thuốc', 'triệu_chứng', 'bác_sĩ', 'xét_nghiệm',
                     'viêm', 'đau', 'sốt', 'huyết', 'tim', 'gan']
    available = [w for w in medical_words if w in model.wv]

    if len(available) < 3:
        print("  [SKIP] Không đủ từ y tế trong vocab để vẽ heatmap")
        return

    vectors = np.array([model.wv[w] for w in available])
    sim_matrix = cosine_similarity(vectors)

    fig, ax = plt.subplots(figsize=(10, 8))
    im = ax.imshow(sim_matrix, cmap='RdYlGn', vmin=0, vmax=1)
    ax.set_xticks(range(len(available)))
    ax.set_yticks(range(len(available)))
    ax.set_xticklabels(available, rotation=45, ha='right', fontsize=10)
    ax.set_yticklabels(available, fontsize=10)

    for i in range(len(available)):
        for j in range(len(available)):
            ax.text(j, i, f'{sim_matrix[i, j]:.2f}', ha='center', va='center', fontsize=8)

    plt.colorbar(im)
    plt.title('Cosine Similarity giữa các từ y tế (Word2Vec)', fontsize=14)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  ✓ Đã lưu: {os.path.basename(output_path)}")


def plot_tsne(model, output_path, top_n=300):
    """Vẽ t-SNE visualization cho top-N từ."""
    words = [w for w, _ in model.wv.most_similar(positive=[], topn=0)][:top_n]
    if len(words) < top_n:
        words = list(model.wv.key_to_index.keys())[:top_n]

    vectors = np.array([model.wv[w] for w in words])

    print(f"  Đang chạy t-SNE cho {len(words)} từ...")
    tsne = TSNE(n_components=2, random_state=42, perplexity=30, max_iter=1000)
    coords = tsne.fit_transform(vectors)

    fig, ax = plt.subplots(figsize=(14, 10))
    ax.scatter(coords[:, 0], coords[:, 1], s=15, alpha=0.6, c='steelblue')

    # Chỉ label một số từ quan trọng
    label_indices = list(range(0, min(80, len(words)), 1))
    for i in label_indices:
        ax.annotate(words[i], (coords[i, 0], coords[i, 1]),
                    fontsize=7, alpha=0.7)

    ax.set_xlabel('t-SNE dim 1')
    ax.set_ylabel('t-SNE dim 2')
    ax.set_title(f'Word2Vec — t-SNE visualization (top-{len(words)} từ)', fontsize=14)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  ✓ Đã lưu: {os.path.basename(output_path)}")


# ============================================================
# 2. TF-IDF
# ============================================================

def compute_tfidf(data):
    """Tính TF-IDF cho toàn bộ corpus."""
    print(f"\n{'=' * 60}")
    print("TF-IDF Vectorizer")
    print(f"{'=' * 60}")

    corpus = [record['cleaned_text'] for record in data]

    vectorizer = TfidfVectorizer(
        max_features=TFIDF_MAX_FEATURES,
        min_df=TFIDF_MIN_DF,
        max_df=TFIDF_MAX_DF,
    )
    tfidf_matrix = vectorizer.fit_transform(corpus)
    feature_names = vectorizer.get_feature_names_out()

    print(f"  Shape: {tfidf_matrix.shape}")
    print(f"  Sparsity: {(1 - tfidf_matrix.nnz / (tfidf_matrix.shape[0] * tfidf_matrix.shape[1])) * 100:.1f}%")

    # Top IDF words
    idf_scores = vectorizer.idf_
    top_idf_indices = np.argsort(idf_scores)[-20:][::-1]
    print(f"\n  Top 20 từ có IDF cao nhất (đặc trưng nhất):")
    for i, idx in enumerate(top_idf_indices, 1):
        print(f"    {i:2}. {feature_names[idx]:<25s} IDF={idf_scores[idx]:.3f}")

    low_idf_indices = np.argsort(idf_scores)[:20]
    print(f"\n  Top 20 từ có IDF thấp nhất (phổ biến nhất):")
    for i, idx in enumerate(low_idf_indices, 1):
        print(f"    {i:2}. {feature_names[idx]:<25s} IDF={idf_scores[idx]:.3f}")

    return tfidf_matrix, vectorizer


def plot_tfidf(tfidf_matrix, vectorizer, output_path):
    """Vẽ biểu đồ phân phối IDF & cosine similarity."""
    feature_names = vectorizer.get_feature_names_out()
    idf_scores = vectorizer.idf_

    fig, axes = plt.subplots(1, 3, figsize=(20, 6))

    # 1. Phân phối IDF
    axes[0].hist(idf_scores, bins=50, color='crimson', alpha=0.7, edgecolor='darkred')
    axes[0].set_xlabel('IDF')
    axes[0].set_ylabel('Số từ')
    axes[0].set_title('Phân phối giá trị IDF')

    # 2. Top 20 IDF cao nhất
    top_idx = np.argsort(idf_scores)[-20:][::-1]
    top_words = [feature_names[i] for i in top_idx]
    top_values = [idf_scores[i] for i in top_idx]
    axes[1].barh(range(len(top_words)), top_values, color='crimson', alpha=0.8)
    axes[1].set_yticks(range(len(top_words)))
    axes[1].set_yticklabels(top_words, fontsize=8)
    axes[1].set_xlabel('IDF')
    axes[1].set_title('Top 20 từ IDF cao nhất (đặc trưng)')
    axes[1].invert_yaxis()

    # 3. Cosine similarity (100 docs đầu)
    n = min(100, tfidf_matrix.shape[0])
    sim = cosine_similarity(tfidf_matrix[:n])
    im = axes[2].imshow(sim, cmap='hot', aspect='auto')
    axes[2].set_xlabel('Tài liệu')
    axes[2].set_ylabel('Tài liệu')
    axes[2].set_title(f'Cosine Similarity ({n} tài liệu đầu) — TF-IDF')
    plt.colorbar(im, ax=axes[2])

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  ✓ Đã lưu: {os.path.basename(output_path)}")


# ============================================================
# 3. DOCUMENT EMBEDDING (Word2Vec Averaging)
# ============================================================

def compute_document_vectors(data, model):
    """
    Tính vector đại diện cho mỗi tài liệu bằng trung bình Word2Vec.
    Phương pháp: Lấy trung bình vector của tất cả các từ trong tài liệu.
    """
    print(f"\n{'=' * 60}")
    print("Document Embedding (Word2Vec Averaging)")
    print(f"{'=' * 60}")

    doc_vectors = []
    valid_indices = []

    for idx, record in enumerate(data):
        tokens = record['cleaned_text'].split()
        word_vectors = [model.wv[w] for w in tokens if w in model.wv]

        if word_vectors:
            doc_vec = np.mean(word_vectors, axis=0)
            doc_vectors.append(doc_vec)
            valid_indices.append(idx)

    doc_matrix = np.array(doc_vectors)
    print(f"  ✓ Document vectors shape: {doc_matrix.shape}")
    print(f"  ✓ Số tài liệu có vector: {len(valid_indices)}/{len(data)}")

    return doc_matrix, valid_indices


def plot_document_similarity(doc_matrix, output_path, n=100):
    """Vẽ cosine similarity giữa các document vectors."""
    n = min(n, doc_matrix.shape[0])
    sim = cosine_similarity(doc_matrix[:n])

    fig, ax = plt.subplots(figsize=(10, 8))
    im = ax.imshow(sim, cmap='viridis', aspect='auto')
    ax.set_xlabel('Tài liệu')
    ax.set_ylabel('Tài liệu')
    ax.set_title(f'Cosine Similarity ({n} tài liệu) — Document Embedding', fontsize=14)
    plt.colorbar(im)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  ✓ Đã lưu: {os.path.basename(output_path)}")


def plot_ngrams(data, output_path):
    """Vẽ biểu đồ Top N-grams (Unigram, Bigram, Trigram)."""
    all_tokens = []
    for record in data:
        all_tokens.extend(record['cleaned_text'].split())

    # Unigram
    unigram_freq = Counter(all_tokens)

    # Bigram
    bigram_freq = Counter()
    for record in data:
        tokens = record['cleaned_text'].split()
        for i in range(len(tokens) - 1):
            bigram_freq[(tokens[i], tokens[i + 1])] += 1

    # Trigram
    trigram_freq = Counter()
    for record in data:
        tokens = record['cleaned_text'].split()
        for i in range(len(tokens) - 2):
            trigram_freq[(tokens[i], tokens[i + 1], tokens[i + 2])] += 1

    fig, axes = plt.subplots(1, 3, figsize=(22, 6))
    top_n = 10

    # Unigram
    top_uni = unigram_freq.most_common(top_n)
    words_u, counts_u = zip(*top_uni)
    axes[0].barh(range(len(words_u)), counts_u, color='dodgerblue')
    axes[0].set_yticks(range(len(words_u)))
    axes[0].set_yticklabels(words_u)
    axes[0].set_xlabel('Tần số')
    axes[0].set_title(f'Top {top_n} Unigram')
    axes[0].invert_yaxis()

    # Bigram
    top_bi = bigram_freq.most_common(top_n)
    words_b = [' '.join(pair) for pair, _ in top_bi]
    counts_b = [c for _, c in top_bi]
    axes[1].barh(range(len(words_b)), counts_b, color='forestgreen')
    axes[1].set_yticks(range(len(words_b)))
    axes[1].set_yticklabels(words_b, fontsize=9)
    axes[1].set_xlabel('Tần số')
    axes[1].set_title(f'Top {top_n} Bigram')
    axes[1].invert_yaxis()

    # Trigram
    top_tri = trigram_freq.most_common(top_n)
    words_t = [' '.join(tri) for tri, _ in top_tri]
    counts_t = [c for _, c in top_tri]
    axes[2].barh(range(len(words_t)), counts_t, color='darkorange')
    axes[2].set_yticks(range(len(words_t)))
    axes[2].set_yticklabels(words_t, fontsize=9)
    axes[2].set_xlabel('Tần số')
    axes[2].set_title(f'Top {top_n} Trigram')
    axes[2].invert_yaxis()

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  ✓ Đã lưu: {os.path.basename(output_path)}")


# ============================================================
# SAVE / EXPORT
# ============================================================

def save_word2vec_model(model, path):
    """Lưu mô hình Word2Vec."""
    model.save(path)
    print(f"  ✓ Đã lưu model: {os.path.basename(path)}")


def save_document_vectors(doc_matrix, valid_indices, data, path):
    """Lưu document vectors dạng JSON (id + vector)."""
    records = []
    for i, idx in enumerate(valid_indices):
        records.append({
            'id': data[idx]['id'],
            'domain': data[idx].get('domain', ''),
            'source': data[idx].get('source', ''),
            'vector': doc_matrix[i].tolist(),
        })
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(records, f, ensure_ascii=False)
    print(f"  ✓ Đã lưu document vectors: {os.path.basename(path)}")


# ============================================================
# MAIN
# ============================================================

def main():
    print("=" * 70)
    print("   TRÍCH XUẤT ĐẶC TRƯNG VĂN BẢN (Word Embedding)")
    print("=" * 70)

    # ── Đọc dữ liệu ──
    data = load_cleaned_data(INPUT_FILE)
    sentences = prepare_sentences(data)

    # Thống kê cơ bản
    all_tokens = [t for s in sentences for t in s]
    token_freq = Counter(all_tokens)
    print(f"\n  Tổng tài liệu  : {len(data):,}")
    print(f"  Tổng từ         : {len(all_tokens):,}")
    print(f"  Từ duy nhất     : {len(token_freq):,}")
    print(f"  TB từ/tài liệu  : {len(all_tokens) / len(data):.0f}")

    # ── 1. Word2Vec Skip-gram ──
    w2v_sg = train_word2vec(sentences, sg=1)
    explore_word2vec(w2v_sg, "Word2Vec Skip-gram")

    # Vẽ biểu đồ Word2Vec
    print("\n  Vẽ biểu đồ Word2Vec...")
    plot_word2vec_similarity(w2v_sg, os.path.join(OUTPUT_DIR, 'w2v_similarity_heatmap.png'))
    plot_tsne(w2v_sg, os.path.join(OUTPUT_DIR, 'w2v_tsne.png'), top_n=300)

    # Lưu model
    save_word2vec_model(w2v_sg, os.path.join(OUTPUT_DIR, 'word2vec_skipgram.model'))

    # ── 2. Word2Vec CBOW ──
    w2v_cbow = train_word2vec(sentences, sg=0)
    explore_word2vec(w2v_cbow, "Word2Vec CBOW")
    save_word2vec_model(w2v_cbow, os.path.join(OUTPUT_DIR, 'word2vec_cbow.model'))

    # ── 3. TF-IDF ──
    tfidf_matrix, tfidf_vectorizer = compute_tfidf(data)
    plot_tfidf(tfidf_matrix, tfidf_vectorizer, os.path.join(OUTPUT_DIR, 'tfidf_result.png'))

    # ── 4. Document Embedding ──
    doc_matrix, valid_indices = compute_document_vectors(data, w2v_sg)
    plot_document_similarity(doc_matrix, os.path.join(OUTPUT_DIR, 'doc_embedding_similarity.png'))
    save_document_vectors(doc_matrix, valid_indices, data, os.path.join(OUTPUT_DIR, 'document_vectors.json'))

    # ── 5. N-grams ──
    print(f"\n{'=' * 60}")
    print("N-gram Analysis")
    print(f"{'=' * 60}")
    plot_ngrams(data, os.path.join(OUTPUT_DIR, 'ngram_result.png'))

    # ── Tổng kết ──
    print(f"\n{'=' * 70}")
    print("TỔNG KẾT")
    print(f"{'=' * 70}")
    print(f"  Word2Vec Skip-gram vocab : {len(w2v_sg.wv):,} từ × {W2V_VECTOR_SIZE}D")
    print(f"  Word2Vec CBOW vocab      : {len(w2v_cbow.wv):,} từ × {W2V_VECTOR_SIZE}D")
    print(f"  TF-IDF matrix            : {tfidf_matrix.shape}")
    print(f"  Document vectors         : {doc_matrix.shape}")

    print(f"\nOutput files:")
    outputs = [
        'word2vec_skipgram.model',
        'word2vec_cbow.model',
        'document_vectors.json',
        'w2v_similarity_heatmap.png',
        'w2v_tsne.png',
        'tfidf_result.png',
        'doc_embedding_similarity.png',
        'ngram_result.png',
    ]
    for f in outputs:
        print(f"  → {f}")

    print("\n✓ HOÀN THÀNH!")


if __name__ == "__main__":
    main()
