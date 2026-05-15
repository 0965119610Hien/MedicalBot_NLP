"""
Medical Hybrid Search System - Production Grade
Combines Vector Search (E5 embeddings) + BM25 Keyword Search with RRF (Reciprocal Rank Fusion)

"""

import json
import logging
import pickle
import re
from pathlib import Path
import sys
from typing import Dict, List, Tuple, Optional
from collections import defaultdict

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

try:
    from project_config import get_config
    _CONFIG = get_config()
    _DEFAULT_OUTPUT_DIR = str(_CONFIG.paths.search_index_dir)
    _DEFAULT_ABBREV = str(_CONFIG.paths.embedding_abbrev_map)
except Exception:
    _CONFIG = None
    _DEFAULT_OUTPUT_DIR = "./medical_search_index"
    _DEFAULT_ABBREV = None

import chromadb
from chromadb.config import Settings
import os
import torch
from sentence_transformers import SentenceTransformer
from rank_bm25 import BM25Okapi
from pyvi import ViTokenizer

try:
    from query_optimizer import replace_abbreviations
except ImportError:
    # Fallback if query_optimizer not available
    def replace_abbreviations(text: str, abbrev_map: Dict) -> str:
        if not abbrev_map:
            return text
        escaped = [re.escape(a) for a in abbrev_map.keys()]
        pattern = re.compile(r'\b(' + '|'.join(escaped) + r')\b', re.IGNORECASE)
        return pattern.sub(lambda m: abbrev_map.get(m.group(0).lower(), m.group(0)), text)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class MedicalHybridSearch:
    """
    Production-grade Hybrid RAG Search System for Vietnamese Medical Domain.
    
    Combines:
    1. Vector Search: E5 multilingual embeddings (intfloat/multilingual-e5-base)
    2. BM25 Keyword Search: Vietnamese-aware tokenization with ViTokenizer
    3. Fusion Strategy: RRF (Reciprocal Rank Fusion) algorithm
    
    Key Design Decisions:
    - E5 requires "passage: " prefix for documents and "query: " for queries
    - Vietnamese compound words need ViTokenizer for proper BM25 tokenization
    - RRF formula: score = 1 / (rank + 60) prevents score explosion at rank 1
    - Abbreviation normalization ensures medical term consistency
    """
    
    def __init__(self, output_dir: Optional[str] = None, abbreviation_file: Optional[str] = None):
        """
        Initialize Medical Hybrid Search system.
        
        Args:
            output_dir: Directory to store ChromaDB and BM25 index files
            abbreviation_file: Path to abbreviation map file (e.g., abbreviation_map.txt)
        """
        resolved_output_dir = output_dir or _DEFAULT_OUTPUT_DIR
        self.output_dir = Path(resolved_output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.chroma_dir = self.output_dir / "chroma_db"
        self.bm25_pkl_path = self.output_dir / "bm25_index.pkl"
        self.metadata_json_path = self.output_dir / "chunk_metadata.json"
        self.abbreviation_map_path = self.output_dir / "abbreviation_map.json"
        
        # Initialize models (lazy loading)
        self.embedding_model = None
        self.chroma_client = None
        self.bm25_index = None
        self.chunk_metadata = {}
        if abbreviation_file is None and _DEFAULT_ABBREV and Path(_DEFAULT_ABBREV).exists():
            abbreviation_file = _DEFAULT_ABBREV
        self.abbreviation_map = self._load_abbreviation_map(abbreviation_file)
        
        # Pre-compile regex patterns for abbreviation replacement (massive performance boost)
        self._compile_abbrev_patterns()
        
        logger.info(f"Initialized MedicalHybridSearch with output_dir: {output_dir}")
        logger.info(f"Loaded {len(self.abbreviation_map)} abbreviations from {abbreviation_file or 'default'}")
    
    def _load_abbreviation_map(self, abbreviation_file: str = None) -> Dict[str, str]:
        """
        Load medical abbreviation mapping from file.
        
        File format:
        # Comment lines start with #
        abbrev=full_term
        bs=bác sĩ
        bn=bệnh nhân
        
        Args:
            abbreviation_file: Path to abbreviation map file
            
        Returns:
            Dictionary mapping abbreviations to full terms
        """
        abbreviation_map = {}
        
        if abbreviation_file and Path(abbreviation_file).exists():
            logger.info(f"Loading abbreviation map from: {abbreviation_file}")
            try:
                with open(abbreviation_file, 'r', encoding='utf-8') as f:
                    for line in f:
                        line = line.strip()
                        # Skip empty lines and comments
                        if not line or line.startswith('#'):
                            continue
                        
                        # Parse abbreviation=expansion format
                        if '=' in line:
                            abbrev, expansion = line.split('=', 1)
                            abbrev = abbrev.strip().lower()
                            expansion = expansion.strip()
                            abbreviation_map[abbrev] = expansion
                
                logger.info(f"Successfully loaded {len(abbreviation_map)} abbreviations")
            except Exception as e:
                logger.error(f"Error loading abbreviation file: {e}")
                logger.info("Using empty abbreviation map")
        else:
            if abbreviation_file:
                logger.warning(f"Abbreviation file not found: {abbreviation_file}")
            logger.info("Using empty abbreviation map")
        
        return abbreviation_map
    
    def _compile_abbrev_patterns(self):
        """
        Pre-compile regex patterns for abbreviation replacement.
        This massive optimization avoids recompiling patterns for each of 66k chunks.
        Creates combined pattern to replace all abbreviations in a single pass.
        """
        if not self.abbreviation_map:
            self.abbrev_pattern = None
            self.abbrev_replacements = {}
            return
        
        # Build single combined regex: (abbrev1|abbrev2|abbrev3|...)
        escaped_abbrevs = [re.escape(abbrev) for abbrev in self.abbreviation_map.keys()]
        pattern_str = r'\b(' + '|'.join(escaped_abbrevs) + r')\b'
        self.abbrev_pattern = re.compile(pattern_str, re.IGNORECASE)
        self.abbrev_replacements = self.abbreviation_map
        
        logger.info(f"Pre-compiled {len(self.abbreviation_map)} abbreviation patterns for fast replacement")
    
    def _normalize_for_embedding(self, text: str) -> str:
        """
        Normalize text for E5 embedding.
        
        Strategy:
        - Replace abbreviations with full terms
        - Keep all punctuation (important for semantic understanding)
        - Preserve original spacing and formatting
        
        Args:
            text: Raw text to normalize
            
        Returns:
            Normalized text ready for E5 embedding
        """
        # Replace abbreviations using pre-compiled pattern (single pass)
        if self.abbrev_pattern:
            text = self.abbrev_pattern.sub(
                lambda m: self.abbrev_replacements[m.group(0).lower()],
                text
            )
        
        # Normalize whitespace but preserve punctuation
        text = ' '.join(text.split())
        
        return text
    
    def _normalize_for_bm25(self, text: str) -> str:
        """
        Normalize text for BM25 indexing.
        
        Strategy:
        - Replace abbreviations with full terms
        - Convert to lowercase
        - Remove punctuation and special characters
        - Preserve word boundaries for tokenization
        
        Args:
            text: Raw text to normalize
            
        Returns:
            Normalized text ready for BM25 preprocessing
        """
        # Replace abbreviations using pre-compiled pattern (single pass)
        if self.abbrev_pattern:
            text = self.abbrev_pattern.sub(
                lambda m: self.abbrev_replacements[m.group(0).lower()],
                text
            )
        
        # Convert to lowercase
        text = text.lower()
        
        # Remove punctuation and special characters (keep spaces for tokenization)
        text = re.sub(r'[^\w\s]', ' ', text)
        
        # Normalize whitespace
        text = ' '.join(text.split())
        
        return text
    
    def _load_embedding_model(self):
        """
        Lazy load E5 embedding model.
        
        Using intfloat/multilingual-e5-base for Vietnamese support.
        This is a high-quality, multilingual model optimized for semantic search.
        """
        if self.embedding_model is None:
            # Enforce GPU-only execution
            try:
                if not torch.cuda.is_available():
                    raise RuntimeError(
                        "CUDA is not available. This pipeline requires a GPU-enabled Python environment. "
                        "Install PyTorch with CUDA and run using that interpreter."
                    )
            except Exception as e:
                # If torch is not importable or any error occurred, raise explicit error
                raise RuntimeError("PyTorch with CUDA support is required to run embeddings on GPU: %s" % e)

            logger.info("Loading E5 embedding model on CUDA: intfloat/multilingual-e5-base")
            # In recent sentence-transformers versions, device can be passed directly
            self.embedding_model = SentenceTransformer('intfloat/multilingual-e5-base', device='cuda')
            # As extra safety, move model to CUDA if available
            try:
                self.embedding_model.to('cuda')
            except Exception:
                pass
            logger.info("E5 model loaded successfully on CUDA")
    
    def _load_chroma_client(self):
        """
        Lazy load ChromaDB persistent client.
        """
        if self.chroma_client is None:
            # Disable telemetry to avoid posthog compatibility issues with older Python versions
            os.environ['CHROMA_TELEMETRY_ENABLED'] = 'false'
            logger.info(f"Initializing ChromaDB client with persistent storage at: {self.chroma_dir}")
            settings = Settings(
                anonymized_telemetry=False,
                allow_reset=True
            )
            self.chroma_client = chromadb.PersistentClient(path=str(self.chroma_dir), settings=settings)
            logger.info("ChromaDB client initialized")
    
    def build_index(self, input_json_path: str):
        """
        Build both Vector and BM25 indexes from processed chunks.
        
        Process Flow:
        1. Load processed_chunks.json
        2. Normalize text for both embedding and BM25
        3. Create E5 embeddings with "passage: " prefix (critical for E5 model)
        4. Create BM25 index with Vietnamese tokenization
        5. Persist ChromaDB and BM25 pickle file
        
        Args:
            input_json_path: Path to processed_chunks.json file
            
        Raises:
            FileNotFoundError: If input JSON file doesn't exist
            ValueError: If JSON format is invalid
        """
        input_path = Path(input_json_path)
        
        if not input_path.exists():
            raise FileNotFoundError(f"Input file not found: {input_json_path}")
        
        logger.info(f"Loading chunks from: {input_json_path}")
        
        try:
            with open(input_path, 'r', encoding='utf-8') as f:
                chunks = json.load(f)
        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid JSON format: {e}")
        
        if not chunks:
            raise ValueError("Input JSON file is empty")

        # Optional quick-run limiter for debugging: set EMBED_MAX_CHUNKS to a positive integer.
        max_chunks_env = os.getenv("EMBED_MAX_CHUNKS", "").strip()
        if max_chunks_env:
            try:
                max_chunks = int(max_chunks_env)
                if max_chunks > 0:
                    original_len = len(chunks)
                    chunks = chunks[:max_chunks]
                    logger.info(f"EMBED_MAX_CHUNKS enabled: using {len(chunks)}/{original_len} chunks")
            except ValueError:
                logger.warning(f"Invalid EMBED_MAX_CHUNKS value: {max_chunks_env}; ignoring")
        
        logger.info(f"Loaded {len(chunks)} chunks")
        
        # Initialize models
        self._load_embedding_model()
        self._load_chroma_client()
        
        # Prepare data for indexing
        chunk_ids = []
        texts_for_embedding = []  # Will have "passage: " prefix
        texts_for_bm25 = []
        metadatas = []
        
        logger.info("Preparing data for indexing...")
        for i, chunk in enumerate(chunks):
            if i % 10000 == 0 and i > 0:
                logger.info(f"  Progress: {i}/{len(chunks)} chunks prepared...")
            
            chunk_id = chunk.get('chunk_id', f'chunk_{i}')
            page_content = chunk.get('page_content', '')
            
            # Store metadata
            metadata = {
                'chunk_id': chunk_id,
                'source_id': chunk.get('source_id', ''),
                'title': chunk.get('title', ''),
                'heading': chunk.get('heading', ''),
                'page_content': page_content,
            }
            
            # Normalize for embedding (keep punctuation)
            text_for_embedding = self._normalize_for_embedding(page_content)
            
            # Normalize for BM25 (remove punctuation, lowercase)
            text_for_bm25 = self._normalize_for_bm25(page_content)
            
            chunk_ids.append(chunk_id)
            # E5 REQUIREMENT: Add "passage: " prefix for document encoding
            texts_for_embedding.append(f"passage: {text_for_embedding}")
            texts_for_bm25.append(text_for_bm25)
            metadatas.append(metadata)
            self.chunk_metadata[chunk_id] = metadata
        
        logger.info(f"Prepared {len(chunk_ids)} chunks for indexing")
        
        # Store in ChromaDB
        logger.info("Encoding + storing embeddings in ChromaDB (streaming batches)...")

        # Recreate the collection on every full rebuild so old indexes with a
        # different embedding dimension do not conflict with the current model.
        try:
            self.chroma_client.delete_collection(name="medical_chunks")
            logger.info("Removed existing ChromaDB collection: medical_chunks")
        except Exception:
            pass

        collection = self.chroma_client.get_or_create_collection(
            name="medical_chunks",
            metadata={"hnsw:space": "cosine"}
        )

        # Encode and upsert per batch to avoid holding all embeddings in RAM
        batch_size = 128
        total = len(chunk_ids)
        for i in range(0, total, batch_size):
            end_idx = min(i + batch_size, total)
            batch_no = (i // batch_size) + 1
            batch_ids = chunk_ids[i:end_idx]
            batch_metadatas = metadatas[i:end_idx]
            batch_texts = texts_for_embedding[i:end_idx]

            batch_embeddings = self.embedding_model.encode(
                batch_texts,
                normalize_embeddings=True,
                show_progress_bar=False,
                batch_size=batch_size,
                device='cuda'
            ).tolist()

            try:
                collection.upsert(
                    ids=batch_ids,
                    embeddings=batch_embeddings,
                    metadatas=batch_metadatas,
                    documents=batch_texts
                )
            except Exception as e:
                logger.exception(f"Upsert failed at batch {batch_no} ({i+1}-{end_idx}/{total}): {e}")
                raise

            if batch_no % 5 == 0 or end_idx == total:
                # Count query is expensive; run periodically for explicit confirmation.
                current_count = collection.count()
                logger.info(
                    f"  Batch {batch_no}: {i+1}-{end_idx}/{total} encoded+stored; "
                    f"collection_count={current_count}"
                )

        logger.info("ChromaDB index created successfully")
        
        # Create BM25 index with Vietnamese tokenization
        logger.info("Building BM25 index...")
        
        # Tokenize using ViTokenizer for proper Vietnamese word segmentation
        tokenized_texts = []
        for text in texts_for_bm25:
            # ViTokenizer.tokenize() returns space-separated tokens
            tokens = ViTokenizer.tokenize(text).split()
            tokenized_texts.append(tokens)
        
        # Create BM25 index
        self.bm25_index = BM25Okapi(tokenized_texts)
        logger.info(f"BM25 index created with {len(tokenized_texts)} documents")
        
        # Save BM25 index to pickle file
        logger.info(f"Saving BM25 index to: {self.bm25_pkl_path}")
        with open(self.bm25_pkl_path, 'wb') as f:
            pickle.dump(self.bm25_index, f)
        
        # Save metadata and abbreviation map
        logger.info(f"Saving metadata to: {self.metadata_json_path}")
        with open(self.metadata_json_path, 'w', encoding='utf-8') as f:
            json.dump(self.chunk_metadata, f, ensure_ascii=False, indent=2)
        
        logger.info(f"Saving abbreviation map to: {self.abbreviation_map_path}")
        with open(self.abbreviation_map_path, 'w', encoding='utf-8') as f:
            json.dump(self.abbreviation_map, f, ensure_ascii=False, indent=2)
        
        logger.info("=" * 80)
        logger.info("INDEX BUILD COMPLETE")
        logger.info(f"  - ChromaDB location: {self.chroma_dir}")
        logger.info(f"  - BM25 pickle location: {self.bm25_pkl_path}")
        logger.info(f"  - Total chunks indexed: {len(chunk_ids)}")
        logger.info("=" * 80)
    
    def load_index(self):
        """
        Load existing indexes from disk.
        
        Raises:
            FileNotFoundError: If index files don't exist
        """
        if not self.bm25_pkl_path.exists():
            raise FileNotFoundError(f"BM25 index not found: {self.bm25_pkl_path}")
        
        logger.info("Loading existing indexes...")
        
        # Load E5 model and ChromaDB
        self._load_embedding_model()
        self._load_chroma_client()
        
        # Load BM25 index
        logger.info(f"Loading BM25 index from: {self.bm25_pkl_path}")
        with open(self.bm25_pkl_path, 'rb') as f:
            self.bm25_index = pickle.load(f)
        
        # Load metadata
        if self.metadata_json_path.exists():
            logger.info(f"Loading metadata from: {self.metadata_json_path}")
            with open(self.metadata_json_path, 'r', encoding='utf-8') as f:
                self.chunk_metadata = json.load(f)
        
        logger.info(f"Loaded indexes with {len(self.chunk_metadata)} chunks")
    
    def hybrid_search(self, user_query: str, top_k: int = 5, top_n_per_index: int = 20) -> List[Dict]:
        """
        Perform hybrid search combining vector and BM25 results using RRF.
        
        Algorithm:
        1. Preprocess query for both BM25 and Vector search
        2. Retrieve Top N results from each index
        3. Apply RRF (Reciprocal Rank Fusion) formula: score = 1 / (rank + 60)
        4. Aggregate RRF scores and return Top K results
        
        Why RRF?
        - Combines rankers without score normalization issues
        - 60 is a constant to prevent score explosion at rank 1
        - Robust to missing results (chunk appears in only one index)
        
        Args:
            user_query: User's search query in Vietnamese
            top_k: Number of final results to return
            top_n_per_index: Number of results to retrieve from each index before fusion
            
        Returns:
            List of dictionaries with keys:
            - chunk_id: Unique chunk identifier
            - page_content: Full text content
            - title: Document title
            - heading: Section heading
            - source_id: Source document ID
            - rrf_score: Final RRF score
            - vector_rank: Rank from vector search (None if not in top N)
            - bm25_rank: Rank from BM25 search (None if not in top N)
        """
        if self.bm25_index is None or self.embedding_model is None:
            raise RuntimeError("Indexes not loaded. Call build_index() or load_index() first.")
        
        logger.info(f"Executing hybrid search for query: '{user_query}'")
        
        # ==================== PREPROCESS QUERY ====================
        # STEP 1: Replace abbreviations in the query BEFORE processing
        clean_query = replace_abbreviations(user_query, self.abbreviation_map)
        if clean_query != user_query:
            logger.info(f"After abbreviation replacement: '{clean_query}'")
        
        # For BM25: normalize and tokenize (using clean_query)
        query_for_bm25 = self._normalize_for_bm25(clean_query)
        # Vietnamese tokenization is critical for compound words
        query_tokens_bm25 = ViTokenizer.tokenize(query_for_bm25).split()
        logger.info(f"BM25 query tokens: {query_tokens_bm25}")
        
        # For Vector search: E5 REQUIREMENT - add "query: " prefix (using clean_query)
        query_for_vector = self._normalize_for_embedding(clean_query)
        query_vector_with_prefix = f"query: {query_for_vector}"
        logger.info(f"Vector query (with prefix): '{query_vector_with_prefix}'")
        
        # ==================== VECTOR SEARCH ====================
        logger.info(f"Performing vector search (top {top_n_per_index})...")
        
        vector_ranked = {}
        try:
            query_embedding = self.embedding_model.encode(
                query_vector_with_prefix,
                normalize_embeddings=True
            )
            
            collection = self.chroma_client.get_collection(name="medical_chunks")
            vector_results = collection.query(
                query_embeddings=[query_embedding.tolist()],
                n_results=top_n_per_index
            )
            
            if vector_results['ids'] and len(vector_results['ids']) > 0:
                for rank, chunk_id in enumerate(vector_results['ids'][0]):
                    # RRF formula: score = 1 / (rank + 60)
                    rrf_score = 1.0 / (rank + 60)
                    vector_ranked[chunk_id] = {
                        'rank': rank,
                        'rrf_score': rrf_score,
                        'metadata': vector_results['metadatas'][0][rank] if vector_results['metadatas'] else {}
                    }
            
            logger.info(f"Vector search returned {len(vector_ranked)} results")
            
        except Exception as e:
            logger.warning(f"⚠️ Vector search failed (Chroma error): {repr(e)[:150]}")
            logger.warning("📊 Falling back to BM25 search only")
            vector_ranked = {}  # Empty - will rely on BM25 results
        
        # ==================== BM25 SEARCH ====================
        logger.info(f"Performing BM25 search (top {top_n_per_index})...")
        
        # Get BM25 scores for all documents
        bm25_scores = self.bm25_index.get_scores(query_tokens_bm25)
        
        # Get top N by score
        top_indices = sorted(
            range(len(bm25_scores)),
            key=lambda i: bm25_scores[i],
            reverse=True
        )[:top_n_per_index]
        
        bm25_ranked = {}
        for rank, doc_idx in enumerate(top_indices):
            if doc_idx < len(list(self.chunk_metadata.keys())):
                chunk_id = list(self.chunk_metadata.keys())[doc_idx]
                # RRF formula: score = 1 / (rank + 60)
                rrf_score = 1.0 / (rank + 60)
                bm25_ranked[chunk_id] = {
                    'rank': rank,
                    'rrf_score': rrf_score,
                    'metadata': self.chunk_metadata.get(chunk_id, {})
                }
        
        logger.info(f"BM25 search returned {len(bm25_ranked)} results")
        
        # ==================== RRF FUSION ====================
        logger.info("Fusing results using RRF...")
        
        # Aggregate RRF scores
        rrf_scores = defaultdict(float)
        source_indices = defaultdict(list)  # Track which index contributed
        
        # Add scores from vector search
        for chunk_id, data in vector_ranked.items():
            rrf_scores[chunk_id] += data['rrf_score']
            source_indices[chunk_id].append('vector')
        
        # Add scores from BM25 search
        for chunk_id, data in bm25_ranked.items():
            rrf_scores[chunk_id] += data['rrf_score']
            source_indices[chunk_id].append('bm25')
        
        # Sort by RRF score
        sorted_results = sorted(
            rrf_scores.items(),
            key=lambda x: x[1],
            reverse=True
        )[:top_k]
        
        logger.info(f"RRF fusion returned {len(sorted_results)} top results")
        
        # ==================== FORMAT RESULTS ====================
        final_results = []
        for chunk_id, rrf_score in sorted_results:
            metadata = self.chunk_metadata.get(chunk_id, {})
            
            result = {
                'chunk_id': chunk_id,
                'page_content': metadata.get('page_content', ''),
                'title': metadata.get('title', ''),
                'heading': metadata.get('heading', ''),
                'source_id': metadata.get('source_id', ''),
                'rrf_score': round(rrf_score, 6),
                'vector_rank': vector_ranked[chunk_id]['rank'] if chunk_id in vector_ranked else None,
                'bm25_rank': bm25_ranked[chunk_id]['rank'] if chunk_id in bm25_ranked else None,
                'sources': '+'.join(source_indices[chunk_id])  # e.g., "vector+bm25"
            }
            final_results.append(result)
        
        logger.info(f"Returning {len(final_results)} final results")
        
        return final_results


def main():
    """
    Medical Hybrid Search Complete Pipeline - Auto-detect and process real data.
    """
    logger.info("=" * 80)
    logger.info("MEDICAL HYBRID SEARCH - COMPLETE PIPELINE")
    logger.info("=" * 80)
    
    # Priority: Use real data from CleanData first, fallback to local
    possible_input_files = [
        "../CleanData/processed_chunks.json",  # Priority 1: CleanData folder
        "./processed_chunks.json"               # Priority 2: Local folder
    ]
    
    input_json = None
    for path in possible_input_files:
        if Path(path).exists():
            input_json = path
            logger.info(f"✓ Found processed_chunks.json at: {input_json}")
            # Get file info
            file_size = Path(input_json).stat().st_size / (1024*1024)
            with open(input_json, 'r', encoding='utf-8') as f:
                chunks = json.load(f)
            logger.info(f"  File size: {file_size:.2f} MB, Chunks: {len(chunks)}")
            break

    if input_json is None:
        logger.error("❌ processed_chunks.json not found in any location!")
        return
    
    # Find abbreviation file
    abbreviation_file = None
    possible_paths = [
        "../CleanData/abbreviation_map.txt",  # Priority 1: CleanData folder
        "./abbreviation_map.txt"               # Priority 2: Local folder
    ]
    
    for path in possible_paths:
        if Path(path).exists():
            abbreviation_file = path
            logger.info(f"Found abbreviation file at: {abbreviation_file}")
            break
    
    # Initialize searcher with abbreviation file
    searcher = MedicalHybridSearch(
        output_dir="./medical_search_index",
        abbreviation_file=abbreviation_file
    )
    
    # Check if index already exists
    if searcher.bm25_pkl_path.exists():
        logger.info("\n" + "=" * 80)
        logger.info("LOADING EXISTING INDEX")
        logger.info("=" * 80)
        searcher.load_index()
    else:
        logger.info("\n" + "=" * 80)
        logger.info("BUILDING INDEX FROM CHUNKS")
        logger.info("=" * 80)
        searcher.build_index(input_json)
    
    # Summary
    logger.info("\n" + "=" * 80)
    logger.info("✅ INDEX PIPELINE COMPLETED")
    logger.info("=" * 80)
    with open(input_json, 'r', encoding='utf-8') as f:
        chunks = json.load(f)
    logger.info(f"📂 Input file: {input_json}")
    logger.info(f"📊 Total chunks indexed: {len(chunks)}")
    logger.info(f"💾 Index location: ./medical_search_index/")
    logger.info("=" * 80 + "\n")


if __name__ == "__main__":
    main()
