"""
Hybrid RAG Data Preprocessing Pipeline (Y tế)

Pipeline xử lý dữ liệu y tế gồm 4 bước:
1. Làm sạch nhẹ dữ liệu thô (Light Cleaning)
2. Cắt theo cấu trúc văn bản (Markdown Header Splitting)
3. Cắt đệ quy kiểm soát độ dài (Recursive Character Splitting)
4. Tiêm metadata & làm sạch sâu (Deep Cleaning & Metadata Injection)

"""

import json
import os
import re
from pathlib import Path
from typing import List, Dict, Any, Optional
from dataclasses import dataclass, asdict
import logging

# LangChain imports
from langchain_text_splitters import (
    MarkdownHeaderTextSplitter,
    RecursiveCharacterTextSplitter
)
from langchain_core.documents import Document

try:
    from clean_data_new import load_vinmec_data, load_hellobacsi_data
    from clean_data_new import (
        build_abbreviation_patterns,
        build_compound_phrase_patterns,
        expand_abbreviations,
        mask_personal_information,
        normalize_whitespace,
        read_key_value_mapping,
        read_line_list,
        remove_urls_and_markup,
        split_joined_compound_words,
    )
    HAS_RAW_LOADERS = True
except Exception:
    HAS_RAW_LOADERS = False


PROJECT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
CLEAN_DATA_DIR = os.path.dirname(__file__)
STOPWORDS_PATH = os.path.join(CLEAN_DATA_DIR, 'vietnamese-stopwords.txt')
MEDICAL_COMPOUND_WORDS_PATH = os.path.join(CLEAN_DATA_DIR, 'medical_compound_words.txt')
ABBREVIATION_MAP_PATH = os.path.join(CLEAN_DATA_DIR, 'abbreviation_map.txt')

_CLEANING_RESOURCES = None


# ============================================================
# LOGGING SETUP
# ============================================================

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# ============================================================
# DATA MODELS
# ============================================================

@dataclass
class ProcessedChunk:
    """Model cho mỗi chunk sau khi xử lý."""
    chunk_id: str
    source_id: str
    title: str
    heading: str
    page_content: str
    char_count: int
    word_count: int
    step_origin: str  # "step1", "step2", "step3"


# ============================================================
# STEP 1: MARKDOWN HEADER TEXT SPLITTER
# ============================================================

def split_by_markdown_headers(documents: List[Dict[str, str]]) -> List[Document]:
    """
    Bước 1: Cắt văn bản dựa trên cấu trúc Markdown (Heading 2, Heading 3).
    
    Args:
        documents: List các dict với keys: id, title, text
        
    Returns:
        List of LangChain Document objects với metadata
    """
    logger.info("="*80)
    logger.info("[BƯỚC 1] Markdown Header Text Splitting")
    logger.info("="*80)
    
    # Định nghĩa các heading separator
    headers_to_split_on = [
        ("#", "Heading 1"),
        ("##", "Heading 2"),
        ("###", "Heading 3"),
    ]
    
    markdown_splitter = MarkdownHeaderTextSplitter(
        headers_to_split_on=headers_to_split_on,
        return_each_line=False
    )
    
    all_docs = []
    total_chunks = 0
    
    for doc in documents:
        try:
            source_id = str(doc.get('id', 'unknown'))
            title = doc.get('title', 'Untitled')
            text = doc.get('text', '')
            
            if not text.strip():
                logger.warning(f"  ⚠️  Document {source_id} có text trống")
                continue
            
            # Split by markdown headers
            splits = markdown_splitter.split_text(text)
            
            # Thêm metadata từ source
            for split in splits:
                split.metadata['source_id'] = source_id
                split.metadata['title'] = title
                split.metadata['step'] = 'step1'
                all_docs.append(split)
            
            logger.info(f"  ✓ Doc {source_id}: {len(splits)} chunks từ Markdown splitting")
            total_chunks += len(splits)
            
        except Exception as e:
            logger.error(f"  ✗ Error processing document {source_id}: {str(e)}")
            continue
    
    logger.info(f"✓ Bước 1 hoàn thành: {total_chunks} chunks từ {len(documents)} documents\n")
    return all_docs


def load_raw_documents(input_file: Optional[str] = None) -> List[Dict[str, str]]:
    """
    Nạp dữ liệu đầu vào.

    Nếu có input_file thì đọc file JSON trực tiếp.
    Nếu không có input_file thì tự đọc raw crawl data từ clean_data_new.py.
    """
    if input_file:
        logger.info(f"📂 Đang đọc input file: {input_file}")
        with open(input_file, 'r', encoding='utf-8') as file:
            raw_documents = json.load(file)

        if not isinstance(raw_documents, list):
            raise ValueError("Input JSON phải là một list")

        normalized_documents: List[Dict[str, str]] = []
        for index, record in enumerate(raw_documents, start=1):
            if not isinstance(record, dict):
                continue
            raw_text = record.get('text', record.get('raw_text', ''))
            normalized_documents.append({
                'id': str(record.get('id', index)),
                'title': record.get('title', f'document_{index}'),
                'text': light_clean_raw_text(raw_text),
            })

        logger.info(f"✓ Loaded {len(normalized_documents)} documents từ input file\n")
        return normalized_documents

    if not HAS_RAW_LOADERS:
        raise ImportError(
            "Không import được load_vinmec_data/load_hellobacsi_data từ clean_data_new.py. "
            "Hãy chạy script trong thư mục CleanData hoặc truyền input_file JSON."
        )

    logger.info("📂 Đang nạp raw crawl data từ clean_data_new.py...")
    vinmec_records = load_vinmec_data()
    hellobacsi_records = load_hellobacsi_data()
    combined_records = vinmec_records + hellobacsi_records

    normalized_documents = []
    for index, record in enumerate(combined_records, start=1):
        domain = record.get('domain', 'unknown')
        source = record.get('source', 'unknown')
        raw_text = record.get('raw_text', '')
        title = record.get('title') or f'{domain}/{source}'

        if not raw_text.strip():
            continue

        normalized_documents.append({
            'id': f'{domain}_{source}_{index}',
            'title': title,
            'text': light_clean_raw_text(raw_text),
        })

    logger.info(f"✓ Loaded {len(normalized_documents)} documents từ raw crawl data\n")
    return normalized_documents


def get_cleaning_resources() -> Dict[str, Any]:
    """Load stopwords, abbreviations, and compound phrase patterns once."""
    global _CLEANING_RESOURCES

    if _CLEANING_RESOURCES is not None:
        return _CLEANING_RESOURCES

    abbreviation_map = read_key_value_mapping(ABBREVIATION_MAP_PATH)
    compound_phrases = read_line_list(MEDICAL_COMPOUND_WORDS_PATH)

    _CLEANING_RESOURCES = {
        'abbreviation_patterns': build_abbreviation_patterns(abbreviation_map),
        'compound_patterns': build_compound_phrase_patterns(compound_phrases),
    }
    return _CLEANING_RESOURCES


def convert_html_headings_to_markdown(text: str) -> str:
    """Convert HTML headings to Markdown headings before splitting."""

    def _replace_heading(match: re.Match, prefix: str) -> str:
        heading_text = re.sub(r"<[^>]+>", "", match.group(1)).strip()
        return f"\n{prefix} {heading_text}\n"

    text = re.sub(r"<h1[^>]*>(.*?)</h1>", lambda m: _replace_heading(m, "#"), text, flags=re.IGNORECASE | re.DOTALL)
    text = re.sub(r"<h2[^>]*>(.*?)</h2>", lambda m: _replace_heading(m, "##"), text, flags=re.IGNORECASE | re.DOTALL)
    text = re.sub(r"<h3[^>]*>(.*?)</h3>", lambda m: _replace_heading(m, "###"), text, flags=re.IGNORECASE | re.DOTALL)
    return text


def light_clean_raw_text(text: str) -> str:
    """
    Làm sạch nhẹ dữ liệu thô nhưng vẫn giữ dấu câu và cấu trúc heading.

    Các bước:
    - Chuyển heading HTML sang Markdown
    - Xóa URL/HTML tags rác
    - Mask email/phone
    - Expand abbreviation
    - Tách compound words
    - Chuẩn hóa khoảng trắng
    """
    resources = get_cleaning_resources()

    text = convert_html_headings_to_markdown(text)
    text = remove_urls_and_markup(text)
    text, _ = mask_personal_information(text)
    text = text.lower()
    text = expand_abbreviations(text, resources['abbreviation_patterns'])
    text = split_joined_compound_words(text, resources['compound_patterns'])
    text = normalize_whitespace(text)
    return text


# ============================================================
# STEP 2: RECURSIVE CHARACTER TEXT SPLITTER
# ============================================================

def split_large_chunks_recursively(docs: List[Document]) -> List[Document]:
    """
    Bước 2: Cắt đệ quy các chunk lớn (> 1500 ký tự).
    
    Args:
        docs: List of Document objects từ Bước 1
        
    Returns:
        List of Document objects sau khi cắt đệ quy
    """
    logger.info("="*80)
    logger.info("[BƯỚC 2] Recursive Character Text Splitting")
    logger.info("="*80)
    
    # Cấu hình splitter
    recursive_splitter = RecursiveCharacterTextSplitter(
        separators=["\n\n", "\n", ".", " "],  # Ưu tiên cắt ở đây
        chunk_size=1500,
        chunk_overlap=150,
        length_function=len,
        is_separator_regex=False
    )
    
    all_docs = []
    chunk_threshold = 1500
    chunks_split = 0
    chunks_passed = 0
    
    for doc in docs:
        text = doc.page_content
        char_count = len(text)
        
        if char_count > chunk_threshold:
            # Cần cắt tiếp
            try:
                sub_splits = recursive_splitter.split_text(text)
                for sub_split in sub_splits:
                    new_doc = Document(
                        page_content=sub_split,
                        metadata={**doc.metadata, 'step': 'step2'}
                    )
                    all_docs.append(new_doc)
                
                logger.debug(f"  Split chunk {char_count}→{len(sub_splits)} sub-chunks")
                chunks_split += 1
                
            except Exception as e:
                logger.warning(f"  ⚠️  Error splitting large chunk: {str(e)}, keeping original")
                doc.metadata['step'] = 'step2'
                all_docs.append(doc)
        else:
            # Chunk nhỏ, giữ nguyên
            doc.metadata['step'] = 'step2'
            all_docs.append(doc)
            chunks_passed += 1
    
    logger.info(f"✓ Bước 2 hoàn thành: {chunks_split} chunks cắt, {chunks_passed} chunks giữ nguyên")
    logger.info(f"✓ Total output: {len(all_docs)} chunks\n")
    return all_docs


# ============================================================
# STEP 3: DEEP CLEANING & METADATA INJECTION
# ============================================================

def deep_clean_text(text: str) -> str:

    # 1. Chuyển thành chữ thường
    text = text.lower()
    
    # 2. Xóa các dòng trống liên tiếp
    text = re.sub(r'\n\s*\n+', '\n', text)
    
    # 3. Xóa khoảng trắng thừa (nhưng giữ \n)
    text = re.sub(r'[ \t]+', ' ', text)
    text = re.sub(r' \n', '\n', text)
    text = re.sub(r'\n ', '\n', text)
    
    # 4. Xóa các ký tự vô nghĩa (nhưng GIỮ dấu câu cơ bản)
    # Loại bỏ: @, #, *, ^, ~, `, |, \, {, }, [, ]
    text = re.sub(r'[@#*^~`|\\{}\[\]]', '', text)
    
    # 5. Fix spacing xung quanh dấu câu
    text = re.sub(r'\s+([.,!?;:])', r'\1', text)  # Loại space trước dấu
    text = re.sub(r'([.,!?;:])(?=[a-zA-Z])', r'\1 ', text)    # Giữ 1 space sau dấu
    
    # 6. Trim leading/trailing whitespace
    text = text.strip()
    
    return text


def inject_metadata_and_clean(docs: List[Document]) -> List[ProcessedChunk]:
    """
    Bước 3: Tiêm metadata & làm sạch sâu cho mỗi chunk.
    
    Format: "Chủ đề: {title} | Phần: {heading}\nNội dung: {cleaned_text}"
    
    Args:
        docs: List of Document objects từ Bước 2
        
    Returns:
        List of ProcessedChunk objects
    """
    logger.info("="*80)
    logger.info("[BƯỚC 3] Metadata Injection & Deep Cleaning")
    logger.info("="*80)
    
    processed_chunks = []
    
    for idx, doc in enumerate(docs):
        try:
            # Extract metadata
            source_id = doc.metadata.get('source_id', 'unknown')
            title = doc.metadata.get('title', 'Untitled')
            
            ## Lấy tất cả các cấp Heading có sẵn và nối lại bằng ký tự ">"
            headings = []
            if 'Heading 1' in doc.metadata: headings.append(doc.metadata['Heading 1'])
            if 'Heading 2' in doc.metadata: headings.append(doc.metadata['Heading 2'])
            if 'Heading 3' in doc.metadata: headings.append(doc.metadata['Heading 3'])

            heading = " > ".join(headings) if headings else "Nội dung chính"
            
            # Làm sạch text
            cleaned_text = deep_clean_text(doc.page_content)
            
            # Format page_content với metadata
            formatted_content = (
                f"Chủ đề: {title} | Phần: {heading}\n"
                f"Nội dung: {cleaned_text}"
            )
            
            # Tính toán metrics
            char_count = len(cleaned_text)
            word_count = len(cleaned_text.split())
            
            # Tạo ProcessedChunk
            chunk = ProcessedChunk(
                chunk_id=f"chunk_{source_id}_{idx:04d}",
                source_id=source_id,
                title=title,
                heading=heading,
                page_content=formatted_content,
                char_count=char_count,
                word_count=word_count,
                step_origin=doc.metadata.get('step', 'unknown')
            )
            
            processed_chunks.append(chunk)
            
            if (idx + 1) % 100 == 0:
                logger.info(f"  Processed {idx + 1} chunks...")
            
        except Exception as e:
            logger.error(f"  ✗ Error processing chunk {idx}: {str(e)}")
            continue
    
    logger.info(f"✓ Bước 3 hoàn thành: {len(processed_chunks)} chunks được xử lý\n")
    return processed_chunks


# ============================================================
# MAIN PIPELINE
# ============================================================

def run_preprocessing_pipeline(
    input_file: Optional[str] = None,
    output_file: str = "processed_chunks.json"
) -> List[ProcessedChunk]:
    """
    Chạy toàn bộ pipeline xử lý dữ liệu 3 bước.
    
    Args:
        input_file: Đường dẫn tới file JSON đầu vào, hoặc None để tự đọc raw crawl data
        output_file: Đường dẫn tới file JSON đầu ra (default: processed_chunks.json)
        
    Returns:
        List of ProcessedChunk objects
    """
    logger.info("\n" + "🚀 " * 40)
    logger.info("HYBRID RAG DATA PREPROCESSING PIPELINE - Y TẾ")
    logger.info("🚀 " * 40 + "\n")
    
    # ────────────────────────────────────────
    # Load input data
    # ────────────────────────────────────────
    try:
        raw_documents = load_raw_documents(input_file)
    except Exception as e:
        logger.error(f"✗ Error loading input data: {str(e)}")
        raise
    
    # ────────────────────────────────────────
    # BƯỚC 1: Markdown Header Splitting
    # ────────────────────────────────────────
    try:
        docs_step1 = split_by_markdown_headers(raw_documents)
    except Exception as e:
        logger.error(f"✗ Pipeline failed at Step 1: {str(e)}")
        raise
    
    # ────────────────────────────────────────
    # BƯỚC 2: Recursive Character Splitting
    # ────────────────────────────────────────
    try:
        docs_step2 = split_large_chunks_recursively(docs_step1)
    except Exception as e:
        logger.error(f"✗ Pipeline failed at Step 2: {str(e)}")
        raise
    
    # ────────────────────────────────────────
    # BƯỚC 3: Deep Cleaning & Metadata Injection
    # ────────────────────────────────────────
    try:
        processed_chunks = inject_metadata_and_clean(docs_step2)
    except Exception as e:
        logger.error(f"✗ Pipeline failed at Step 3: {str(e)}")
        raise
    
    # ────────────────────────────────────────
    # Save output
    # ────────────────────────────────────────
    try:
        logger.info(f"💾 Đang lưu output: {output_file}")
        output_data = [asdict(chunk) for chunk in processed_chunks]
        
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(output_data, f, ensure_ascii=False, indent=2)
        
        logger.info(f"✓ Saved {len(processed_chunks)} chunks to {output_file}\n")
    except Exception as e:
        logger.error(f"✗ Error saving output file: {str(e)}")
        raise
    
    # ────────────────────────────────────────
    # Statistics
    # ────────────────────────────────────────
    logger.info("="*80)
    logger.info("📊 THỐNG KÊ")
    logger.info("="*80)
    
    total_chars = sum(c.char_count for c in processed_chunks)
    total_words = sum(c.word_count for c in processed_chunks)
    avg_chars = total_chars / len(processed_chunks) if processed_chunks else 0
    avg_words = total_words / len(processed_chunks) if processed_chunks else 0
    
    logger.info(f"✓ Total chunks: {len(processed_chunks)}")
    logger.info(f"✓ Total characters: {total_chars:,}")
    logger.info(f"✓ Total words: {total_words:,}")
    logger.info(f"✓ Avg chars/chunk: {avg_chars:.1f}")
    logger.info(f"✓ Avg words/chunk: {avg_words:.1f}")
    
    # Distribution by step
    step1_count = sum(1 for c in processed_chunks if c.step_origin == 'step1')
    step2_count = sum(1 for c in processed_chunks if c.step_origin == 'step2')
    
    logger.info(f"\nChunks distribution:")
    logger.info(f"  • From Step 1 (no split): {step1_count}")
    logger.info(f"  • From Step 2 (recursive split): {step2_count}")
    
    logger.info("\n" + "✅ " * 40)
    logger.info("PIPELINE COMPLETED SUCCESSFULLY!")
    logger.info("✅ " * 40 + "\n")
    
    return processed_chunks

def main():
    """
    Hàm main để chạy pipeline với raw crawl data.
    """
    # Chạy pipeline
    try:
        processed_chunks = run_preprocessing_pipeline(
            output_file="processed_chunks.json"
        )
        
        # In mẫu output
        logger.info("\n" + "="*80)
        logger.info("📋 SAMPLE OUTPUT (Chunk 1)")
        logger.info("="*80)
        if processed_chunks:
            first_chunk = processed_chunks[0]
            logger.info(f"Chunk ID: {first_chunk.chunk_id}")
            logger.info(f"Title: {first_chunk.title}")
            logger.info(f"Heading: {first_chunk.heading}")
            logger.info(f"Word count: {first_chunk.word_count}")
            logger.info(f"Char count: {first_chunk.char_count}")
            logger.info(f"\nContent preview (first 300 chars):")
            logger.info(f"{first_chunk.page_content[:300]}...\n")
    
    except Exception as e:
        logger.error(f"✗ Pipeline execution failed: {str(e)}")


if __name__ == "__main__":
    main()
