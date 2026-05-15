from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class ChunkingConfig:
    chunk_size: int = 1500
    chunk_overlap: int = 150
    separators: List[str] = field(default_factory=lambda: ["\n\n", "\n", ".", " "])


@dataclass
class DedupConfig:
    enable_exact: bool = True
    enable_fuzzy: bool = True
    fuzzy_threshold: float = 0.92
    fuzzy_min_words: int = 30


@dataclass
class PipelineConfig:
    input_json: Optional[str] = None
    output_json: str = ""
    stats_json: str = ""
    min_words_per_record: int = 10
    enable_pre_chunk_filter: bool = False
    pre_chunk_min_chars: int = 300
    pre_chunk_max_chars: int = 50000
    pre_chunk_generic_ratio: float = 0.6
    target_records: int = 0
    chunking: ChunkingConfig = field(default_factory=ChunkingConfig)
    dedup: DedupConfig = field(default_factory=DedupConfig)
