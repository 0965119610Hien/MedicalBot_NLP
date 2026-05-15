import argparse
import logging
from pathlib import Path
import sys

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from project_config import get_config
from .config import ChunkingConfig, DedupConfig, PipelineConfig
from .pipeline import run_pipeline

CONFIG = get_config()
DEFAULT_OUTPUT_JSON = str(CONFIG.paths.processed_chunks)
DEFAULT_STATS_JSON = str(CONFIG.paths.processed_stats)
DEFAULT_MIN_WORDS = CONFIG.pipeline.min_words
DEFAULT_CHUNK_SIZE = CONFIG.pipeline.chunk_size
DEFAULT_CHUNK_OVERLAP = CONFIG.pipeline.chunk_overlap


def build_config(args: argparse.Namespace) -> PipelineConfig:
    chunking = ChunkingConfig(
        chunk_size=args.chunk_size,
        chunk_overlap=args.chunk_overlap,
    )
    dedup = DedupConfig(
        enable_exact=not args.no_exact,
        enable_fuzzy=not args.no_fuzzy,
        fuzzy_threshold=args.fuzzy_threshold,
    )

    return PipelineConfig(
        input_json=args.input_json,
        output_json=args.output_json,
        stats_json=args.stats_json,
        min_words_per_record=args.min_words,
        enable_pre_chunk_filter=args.enable_pre_chunk_filter,
        pre_chunk_min_chars=args.pre_chunk_min_chars,
        pre_chunk_max_chars=args.pre_chunk_max_chars,
        pre_chunk_generic_ratio=args.pre_chunk_generic_ratio,
        target_records=args.target_records,
        chunking=chunking,
        dedup=dedup,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Medical data preprocessing pipeline")
    parser.add_argument("--input-json", default=None, help="Optional input JSON file")
    parser.add_argument("--output-json", default=DEFAULT_OUTPUT_JSON, help="Output JSON path")
    parser.add_argument("--stats-json", default=DEFAULT_STATS_JSON, help="Stats JSON path")

    parser.add_argument("--min-words", type=int, default=DEFAULT_MIN_WORDS, help="Minimum words per record")
    parser.add_argument("--chunk-size", type=int, default=DEFAULT_CHUNK_SIZE, help="Chunk size in characters")
    parser.add_argument("--chunk-overlap", type=int, default=DEFAULT_CHUNK_OVERLAP, help="Chunk overlap in characters")

    parser.add_argument("--no-exact", action="store_true", help="Disable exact dedup")
    parser.add_argument("--no-fuzzy", action="store_true", help="Disable fuzzy dedup")
    parser.add_argument("--fuzzy-threshold", type=float, default=0.92, help="Fuzzy dedup threshold")

    parser.add_argument("--enable-pre-chunk-filter", action="store_true", help="Enable pre-chunk filtering")
    parser.add_argument("--pre-chunk-min-chars", type=int, default=300, help="Min chars for pre-chunk filter")
    parser.add_argument("--pre-chunk-max-chars", type=int, default=50000, help="Max chars for pre-chunk filter")
    parser.add_argument("--pre-chunk-generic-ratio", type=float, default=0.6, help="Generic ratio threshold")
    parser.add_argument("--target-records", type=int, default=0, help="Limit records before chunking")

    return parser.parse_args()


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
    args = parse_args()
    config = build_config(args)

    output_json = Path(config.output_json) if config.output_json else "(default)"
    logging.info("Starting pipeline, output=%s", output_json)
    run_pipeline(config)


if __name__ == "__main__":
    main()
