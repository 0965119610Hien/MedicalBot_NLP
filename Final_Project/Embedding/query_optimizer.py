"""
Query Optimizer Module for Medical Hybrid RAG System
- Optimizes user queries using LM Studio (Local Model via OpenAI-compatible API)
- Handles abbreviation expansion and semantic query enhancement
"""

import logging
from pathlib import Path
import re
import sys
from typing import Dict, Optional

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from project_config import get_config

CONFIG = get_config()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# LM Studio configuration
LM_STUDIO_URL = CONFIG.lm_studio.url
LM_STUDIO_MODEL = CONFIG.lm_studio.model


def _normalize_tokens(text: str) -> list:
    cleaned = re.sub(r"[^0-9A-Za-zÀ-ỹ]+", " ", text.lower())
    return [t for t in cleaned.split() if t]


def _is_rewrite_safe(raw_query: str, rewritten_query: str) -> bool:
    if not rewritten_query:
        return False

    bad_phrases = [
        "đồng bộ hóa",
        "đồng bộ",
        "tối ưu hóa",
        "đăng nhập",
        "cài đặt",
    ]
    lowered_rewrite = rewritten_query.lower()
    if any(phrase in lowered_rewrite for phrase in bad_phrases):
        return False

    stopwords = {
        "toi", "tôi", "muon", "muốn", "biet", "biết", "hoi", "hỏi", "ve", "về",
        "la", "là", "cho", "cua", "của", "va", "và", "co", "có", "khong",
        "không", "nhu", "như", "khi", "trong", "duoc", "được", "dung",
        "dùng", "trong", "tren", "em", "be", "bé", "tre", "trẻ", "nguoi", "người"
    }

    raw_tokens = [t for t in _normalize_tokens(raw_query) if t not in stopwords]
    rewrite_tokens = set(_normalize_tokens(rewritten_query))

    if not raw_tokens:
        return True

    overlap = [t for t in raw_tokens if t in rewrite_tokens]
    overlap_ratio = len(overlap) / len(set(raw_tokens))

    # Require at least one meaningful token and decent overlap.
    return len(overlap) >= 1 and overlap_ratio >= 0.4


def rewrite_query_with_lm_studio(raw_query: str, lm_studio_url: str = None) -> str:
    """
    Optimize user query using LM Studio (local Qwen2.5-3B model via OpenAI API).
    
    Replaces abbreviations, expands to medical terminology, and adds synonyms
    for better vector/BM25 search performance.
    
    Args:
        raw_query: Original user query (may contain abbreviations, colloquial terms)
        lm_studio_url: LM Studio server URL (default: http://localhost:1234/v1/chat/completions)
    
    Returns:
        Optimized query string (or raw_query if API fails)
    """
    try:
        import requests
    except ImportError:
        logger.warning("requests library not available; skipping query optimization")
        return raw_query
    
    try:
        url = lm_studio_url or LM_STUDIO_URL
        
        # System prompt: act as a medical expert to rewrite query
        system_prompt = """Bạn là một bác sĩ chuyên khoa y tế.
    Nhận một câu hỏi y tế từ người dùng (có thể viết tắt, dùng từ bình dân).
    Hãy viết lại câu hỏi để tối ưu tìm kiếm, với các yêu cầu BẮT BUỘC:
    1. Giữ NGUYÊN ý nghĩa, không thêm khái niệm mới hoặc hành động mới.
    2. Chỉ mở rộng viết tắt, chuẩn hóa thuật ngữ y khoa, và thêm đồng nghĩa liên quan.
    3. KHÔNG sáng tạo cụm từ lạ hoặc chuyển sang chủ đề khác.
    4. Giữ lại các từ khóa chính (tên thuốc/triệu chứng) trong câu gốc.

    CHỈ trả về 1 câu đã được viết lại, không giải thích thêm."""
        
        # OpenAI-compatible request format for LM Studio
        payload = {
            "model": LM_STUDIO_MODEL,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": raw_query}
            ],
            "temperature": 0.3,
            "max_tokens": 200
        }
        
        logger.info(f"Calling LM Studio at {url} with model {LM_STUDIO_MODEL}")
        response = requests.post(url, json=payload, timeout=15)
        
        if response.status_code == 200:
            result = response.json()
            if "choices" in result and len(result["choices"]) > 0:
                choice = result["choices"][0]
                if "message" in choice and "content" in choice["message"]:
                    text = choice["message"]["content"].strip()
                    if text:
                        logger.info(f"LM Studio response: '{text[:80]}...'")
                        if _is_rewrite_safe(raw_query, text):
                            return text
                        logger.warning("LM Studio rewrite deemed unsafe; using raw query")
                        return raw_query
        else:
            logger.warning(f"LM Studio API error: {response.status_code} - {response.text[:100]}")
        
        return raw_query
        
    except Exception as e:
        logger.warning(f"LM Studio query optimization failed ({type(e).__name__}: {str(e)[:100]})")
        logger.info("Using raw query as fallback")
        return raw_query


def replace_abbreviations(text: str, abbreviation_map: Dict[str, str]) -> str:
    """
    Replace medical abbreviations in text with full terms.
    
    Args:
        text: Input text (may contain abbreviations)
        abbreviation_map: Dict mapping abbreviations (lowercase) to full terms
    
    Returns:
        Text with abbreviations replaced
    """
    if not abbreviation_map:
        return text
    
    # Build regex pattern from all abbreviations (word boundary match)
    escaped_abbrevs = [re.escape(abbrev) for abbrev in abbreviation_map.keys()]
    pattern_str = r'\b(' + '|'.join(escaped_abbrevs) + r')\b'
    pattern = re.compile(pattern_str, re.IGNORECASE)
    
    def replacer(match):
        abbrev_lower = match.group(0).lower()
        return abbreviation_map.get(abbrev_lower, match.group(0))
    
    result = pattern.sub(replacer, text)
    
    if result != text:
        logger.info(f"Abbreviations replaced: '{text}' -> '{result}'")
    
    return result


def optimize_query_pipeline(
    raw_query: str,
    abbreviation_map: Dict[str, str],
    use_lm_studio: bool = True,
    lm_studio_url: Optional[str] = None
) -> str:
    """
    Full query optimization pipeline:
    1. Replace abbreviations (internal)
    2. Rewrite with LM Studio (semantic + query expansion)
    
    Args:
        raw_query: Original query from user
        abbreviation_map: Local abbreviation mappings
        use_lm_studio: Whether to call LM Studio API
        lm_studio_url: Optional LM Studio server URL (default: http://localhost:1234/v1/chat/completions)
    
    Returns:
        Fully optimized query
    """
    logger.info(f"=== Query Optimization Pipeline ===")
    logger.info(f"Step 0 (Raw Query): {raw_query}")
    
    # Step 1: Replace abbreviations locally
    query_after_abbrev = replace_abbreviations(raw_query, abbreviation_map)
    logger.info(f"Step 1 (After Abbreviation Replacement): {query_after_abbrev}")
    
    # Step 2: LM Studio semantic rewrite + query expansion
    if use_lm_studio:
        skip_phrases = [
            "tác dụng phụ",
            "chống chỉ định",
            "tương tác thuốc",
            "liều lượng",
            "liều dùng",
            "cách dùng",
        ]
        lowered_query = query_after_abbrev.lower()
        if query_after_abbrev == raw_query and any(phrase in lowered_query for phrase in skip_phrases):
            logger.info("Skipping LM Studio rewrite to avoid intent drift")
            final_query = query_after_abbrev
        else:
            final_query = rewrite_query_with_lm_studio(query_after_abbrev, lm_studio_url=lm_studio_url)
        logger.info(f"Step 2 (After LM Studio Rewrite): {final_query}")
    else:
        final_query = query_after_abbrev
        logger.info(f"Step 2 (LM Studio skipped): {final_query}")
    
    return final_query


if __name__ == "__main__":
    # Example abbreviation map
    sample_abbrev_map = {
        "bn": "bệnh nhân",
        "tđ": "tiểu đường",
        "bs": "bác sĩ",
        "hạ sốt": "giảm nhiệt độ",  # For semantic expansion
    }
    
    # Test queries
    test_queries = [
        "thuốc giảm nhiệt độ cho bn bị tđ loại 2",
        "bn bị tđ có được uống paracetamol không",
        "hạ sốt cho bé dưới 2 tuổi",
    ]
    
    print("=" * 80)
    print("QUERY OPTIMIZER TEST")
    print("=" * 80)
    
    for query in test_queries:
        print(f"\nRaw query: {query}")
        optimized = optimize_query_pipeline(
            query,
            sample_abbrev_map,
            use_gemini=False  # Set to True if you have API key
        )
        print(f"Optimized: {optimized}")
        print("-" * 80)
