import json
import logging
import re
from collections import Counter
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

from .config import PipelineConfig

try:
    from CleanData import clean_data_new as cdn
except Exception as exc:  # pragma: no cover - import guard
    raise ImportError("CleanData.clean_data_new is required for the pipeline") from exc

LOGGER = logging.getLogger("data_pipeline")

PROJECT_ROOT = Path(__file__).resolve().parents[1]
CLEAN_DATA_DIR = PROJECT_ROOT / "CleanData"

try:
    from project_config import get_config
    _CONFIG = get_config()
    DEFAULT_OUTPUT_JSON = _CONFIG.paths.processed_chunks
    DEFAULT_STATS_JSON = _CONFIG.paths.processed_stats
except Exception:
    DEFAULT_OUTPUT_JSON = CLEAN_DATA_DIR / "processed_chunks.json"
    DEFAULT_STATS_JSON = CLEAN_DATA_DIR / "processed_chunks_stats.json"

HEADING_PATTERN = re.compile(r"^(#{1,3})\s+(.*)$")


@dataclass
class DocSegment:
    page_content: str
    metadata: Dict[str, str]


@dataclass
class ProcessedChunk:
    chunk_id: str
    source_id: str
    title: str
    heading: str
    page_content: str
    char_count: int
    word_count: int
    step_origin: str


def convert_html_headings_to_markdown(text: str) -> str:
    def replace_heading(match: re.Match, prefix: str) -> str:
        heading_text = re.sub(r"<[^>]+>", "", match.group(1)).strip()
        return f"\n{prefix} {heading_text}\n"

    text = re.sub(r"<h1[^>]*>(.*?)</h1>", lambda m: replace_heading(m, "#"), text, flags=re.IGNORECASE | re.DOTALL)
    text = re.sub(r"<h2[^>]*>(.*?)</h2>", lambda m: replace_heading(m, "##"), text, flags=re.IGNORECASE | re.DOTALL)
    text = re.sub(r"<h3[^>]*>(.*?)</h3>", lambda m: replace_heading(m, "###"), text, flags=re.IGNORECASE | re.DOTALL)
    return text


def normalize_whitespace_preserve_newlines(text: str) -> str:
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r" *\n *", "\n", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def load_raw_records(input_json: Optional[str]) -> List[Dict[str, str]]:
    if input_json:
        input_path = Path(input_json)
        if not input_path.exists():
            raise FileNotFoundError(f"Input JSON not found: {input_json}")
        with input_path.open("r", encoding="utf-8") as file:
            records = json.load(file)
        if not isinstance(records, list):
            raise ValueError("Input JSON must be a list of objects")
        return records

    vinmec_records = cdn.load_vinmec_data()
    hellobacsi_records = cdn.load_hellobacsi_data()
    return vinmec_records + hellobacsi_records


def light_clean_text(
    raw_text: str,
    abbreviation_patterns: Sequence[Tuple[re.Pattern, str]],
    compound_patterns: Sequence[Tuple[str, str]],
) -> str:
    text = convert_html_headings_to_markdown(raw_text)
    text = cdn.remove_urls_and_markup(text)
    text, _ = cdn.mask_personal_information(text)
    text = text.lower()
    text = cdn.expand_abbreviations(text, abbreviation_patterns)
    text = cdn.split_joined_compound_words(text, compound_patterns)
    text = normalize_whitespace_preserve_newlines(text)
    return text


def prepare_clean_records(
    raw_records: List[Dict[str, str]],
    config: PipelineConfig,
    abbreviation_patterns: Sequence[Tuple[re.Pattern, str]],
    compound_patterns: Sequence[Tuple[str, str]],
) -> Tuple[List[Dict[str, str]], Dict[str, int]]:
    cleaned_records: List[Dict[str, str]] = []
    stats = Counter()

    for record in raw_records:
        raw_text = record.get("raw_text") or record.get("text") or record.get("content") or ""
        if not isinstance(raw_text, str) or not raw_text.strip():
            stats["dropped_empty"] += 1
            continue

        cleaned_text = light_clean_text(raw_text, abbreviation_patterns, compound_patterns)
        tokens = cleaned_text.split()
        if len(tokens) < config.min_words_per_record:
            stats["dropped_short"] += 1
            continue

        domain = record.get("domain", "unknown")
        source = record.get("source", "unknown")
        title = record.get("title") or record.get("name") or f"{domain}/{source}"

        cleaned_records.append({
            "id": str(len(cleaned_records) + 1),
            "domain": domain,
            "source": source,
            "url": record.get("url", ""),
            "title": title,
            "cleaned_text": cleaned_text,
            "word_count": len(tokens),
        })

    stats["cleaned_records"] = len(cleaned_records)
    return cleaned_records, dict(stats)


def apply_dedup(records: List[Dict[str, str]], config: PipelineConfig) -> Tuple[List[Dict[str, str]], Dict[str, int]]:
    stats: Dict[str, int] = {
        "exact_dedup_removed": 0,
        "fuzzy_dedup_removed": 0,
    }

    if config.dedup.enable_exact:
        records, removed = cdn.deduplicate_exact_records(records, text_field="cleaned_text")
        stats["exact_dedup_removed"] = removed

    if config.dedup.enable_fuzzy:
        records, removed = cdn.deduplicate_fuzzy_records(records, threshold=config.dedup.fuzzy_threshold)
        stats["fuzzy_dedup_removed"] = removed

    for index, record in enumerate(records, start=1):
        record["id"] = str(index)

    return records, stats


def pre_chunk_filter_records(records: List[Dict[str, str]], config: PipelineConfig) -> Tuple[List[Dict[str, str]], Dict[str, int]]:
    if not config.enable_pre_chunk_filter:
        return records, {}

    stats = Counter()
    filtered: List[Dict[str, str]] = []

    for record in records:
        text = record.get("cleaned_text", "")
        if len(text) < config.pre_chunk_min_chars:
            stats["pre_chunk_drop_too_short"] += 1
            continue
        if len(text) > config.pre_chunk_max_chars:
            stats["pre_chunk_drop_too_long"] += 1
            continue
        filtered.append(record)

    generic_filtered: List[Dict[str, str]] = []
    for record in filtered:
        original_text = record.get("cleaned_text", "")
        cleaned_text = cdn.remove_generic_fragments(original_text)
        if not cleaned_text:
            stats["pre_chunk_drop_empty_after_generic"] += 1
            continue
        if len(cleaned_text) < len(original_text) * config.pre_chunk_generic_ratio:
            stats["pre_chunk_drop_generic_heavy"] += 1
            continue
        updated = dict(record)
        updated["cleaned_text"] = cleaned_text
        updated["word_count"] = len(cleaned_text.split())
        generic_filtered.append(updated)

    selected = generic_filtered
    if config.target_records and len(generic_filtered) > config.target_records:
        ranked = sorted(generic_filtered, key=cdn.score_record_for_chunking, reverse=True)
        selected = ranked[: config.target_records]
        stats["pre_chunk_target_selected"] = len(selected)
        stats["pre_chunk_target_dropped"] = len(ranked) - len(selected)

    for index, record in enumerate(selected, start=1):
        record["id"] = str(index)

    stats["pre_chunk_output_records"] = len(selected)
    return selected, dict(stats)


def split_by_markdown_headers(documents: List[Dict[str, str]]) -> List[DocSegment]:
    segments: List[DocSegment] = []

    for doc in documents:
        source_id = str(doc.get("id", "unknown"))
        title = doc.get("title", "Untitled")
        text = doc.get("text", "")
        if not text.strip():
            continue

        heading_1 = None
        heading_2 = None
        heading_3 = None
        current_lines: List[str] = []

        def flush_segment() -> None:
            if not current_lines:
                return
            content = "\n".join(current_lines).strip()
            if not content:
                return
            segments.append(
                DocSegment(
                    page_content=content,
                    metadata={
                        "source_id": source_id,
                        "title": title,
                        "heading_1": heading_1 or "",
                        "heading_2": heading_2 or "",
                        "heading_3": heading_3 or "",
                        "step": "step1",
                    },
                )
            )

        for line in text.splitlines():
            match = HEADING_PATTERN.match(line.strip())
            if match:
                flush_segment()
                level = len(match.group(1))
                heading_text = match.group(2).strip()
                if level == 1:
                    heading_1 = heading_text
                    heading_2 = None
                    heading_3 = None
                elif level == 2:
                    heading_2 = heading_text
                    heading_3 = None
                else:
                    heading_3 = heading_text
                current_lines = []
                continue
            current_lines.append(line)

        flush_segment()

    return segments


def split_with_separator(text: str, separator: str) -> List[str]:
    if separator == "":
        return list(text)
    if separator not in text:
        return [text]
    parts = text.split(separator)
    return [part + separator for part in parts[:-1]] + [parts[-1]]


def split_text_recursive(
    text: str,
    separators: Sequence[str],
    chunk_size: int,
    chunk_overlap: int,
) -> List[str]:
    if len(text) <= chunk_size:
        return [text]
    if not separators:
        chunks = []
        start = 0
        while start < len(text):
            end = min(len(text), start + chunk_size)
            chunks.append(text[start:end])
            if chunk_overlap <= 0:
                start = end
            else:
                start = max(end - chunk_overlap, start + 1)
        return chunks

    separator = separators[0]
    splits = split_with_separator(text, separator)
    chunks: List[str] = []
    buffer = ""

    for part in splits:
        if len(buffer) + len(part) > chunk_size:
            if buffer:
                chunks.append(buffer)
                if chunk_overlap > 0:
                    buffer = buffer[-chunk_overlap:]
                else:
                    buffer = ""
            if len(part) > chunk_size:
                chunks.extend(split_text_recursive(part, separators[1:], chunk_size, chunk_overlap))
                buffer = ""
                continue
        buffer += part

    if buffer:
        chunks.append(buffer)

    return [chunk.strip() for chunk in chunks if chunk.strip()]


def split_large_chunks_recursively(
    docs: List[DocSegment],
    chunk_size: int,
    chunk_overlap: int,
    separators: Sequence[str],
) -> List[DocSegment]:
    output: List[DocSegment] = []

    for doc in docs:
        if len(doc.page_content) <= chunk_size:
            doc.metadata["step"] = "step2"
            output.append(doc)
            continue

        splits = split_text_recursive(doc.page_content, separators, chunk_size, chunk_overlap)
        for split in splits:
            output.append(
                DocSegment(
                    page_content=split,
                    metadata={**doc.metadata, "step": "step2"},
                )
            )

    return output


def deep_clean_text(text: str) -> str:
    text = text.lower()
    text = re.sub(r"\n\s*\n+", "\n", text)
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r" \n", "\n", text)
    text = re.sub(r"\n ", "\n", text)
    text = re.sub(r"[@#*^~`|\\{}\[\]]", "", text)
    text = re.sub(r"\s+([.,!?;:])", r"\1", text)
    text = re.sub(r"([.,!?;:])(?=[a-zA-Z])", r"\1 ", text)
    return text.strip()


def inject_metadata_and_clean(docs: List[DocSegment]) -> List[ProcessedChunk]:
    processed: List[ProcessedChunk] = []

    for idx, doc in enumerate(docs):
        source_id = doc.metadata.get("source_id", "unknown")
        title = doc.metadata.get("title", "Untitled")
        heading_parts = [
            doc.metadata.get("heading_1", ""),
            doc.metadata.get("heading_2", ""),
            doc.metadata.get("heading_3", ""),
        ]
        heading = " > ".join([part for part in heading_parts if part]) or "Nội dung chính"

        cleaned_text = deep_clean_text(doc.page_content)
        formatted_content = (
            f"Chủ đề: {title} | Phần: {heading}\n"
            f"Nội dung: {cleaned_text}"
        )

        processed.append(
            ProcessedChunk(
                chunk_id=f"chunk_{source_id}_{idx:04d}",
                source_id=source_id,
                title=title,
                heading=heading,
                page_content=formatted_content,
                char_count=len(cleaned_text),
                word_count=len(cleaned_text.split()),
                step_origin=doc.metadata.get("step", "unknown"),
            )
        )

    return processed


def run_pipeline(config: PipelineConfig) -> List[ProcessedChunk]:
    output_json = Path(config.output_json) if config.output_json else DEFAULT_OUTPUT_JSON
    stats_json = Path(config.stats_json) if config.stats_json else DEFAULT_STATS_JSON
    output_json.parent.mkdir(parents=True, exist_ok=True)
    stats_json.parent.mkdir(parents=True, exist_ok=True)

    raw_records = load_raw_records(config.input_json)
    stats = {
        "raw_records": len(raw_records),
    }

    stopwords = cdn.read_line_list(str(CLEAN_DATA_DIR / "vietnamese-stopwords.txt"))
    compound_phrases = cdn.read_line_list(str(CLEAN_DATA_DIR / "medical_compound_words.txt"))
    abbreviation_map = cdn.read_key_value_mapping(str(CLEAN_DATA_DIR / "abbreviation_map.txt"))

    abbreviation_patterns = cdn.build_abbreviation_patterns(abbreviation_map)
    compound_patterns = cdn.build_compound_phrase_patterns(compound_phrases)

    cleaned_records, clean_stats = prepare_clean_records(
        raw_records,
        config,
        abbreviation_patterns,
        compound_patterns,
    )
    stats.update(clean_stats)

    if config.enable_pre_chunk_filter:
        cleaned_records, pre_chunk_stats = pre_chunk_filter_records(cleaned_records, config)
        stats.update(pre_chunk_stats)

    cleaned_records, dedup_stats = apply_dedup(cleaned_records, config)
    stats.update(dedup_stats)

    documents = [
        {
            "id": record["id"],
            "title": record.get("title", "Untitled"),
            "text": record.get("cleaned_text", ""),
        }
        for record in cleaned_records
    ]

    step1_docs = split_by_markdown_headers(documents)
    step2_docs = split_large_chunks_recursively(
        step1_docs,
        chunk_size=config.chunking.chunk_size,
        chunk_overlap=config.chunking.chunk_overlap,
        separators=config.chunking.separators,
    )
    processed_chunks = inject_metadata_and_clean(step2_docs)

    stats["processed_chunks"] = len(processed_chunks)
    if processed_chunks:
        stats["avg_chars_per_chunk"] = sum(c.char_count for c in processed_chunks) / len(processed_chunks)
        stats["avg_words_per_chunk"] = sum(c.word_count for c in processed_chunks) / len(processed_chunks)

    with output_json.open("w", encoding="utf-8") as file:
        json.dump([asdict(chunk) for chunk in processed_chunks], file, ensure_ascii=False, indent=2)

    with stats_json.open("w", encoding="utf-8") as file:
        json.dump(stats, file, ensure_ascii=False, indent=2)

    LOGGER.info("Wrote %s chunks to %s", len(processed_chunks), output_json)
    LOGGER.info("Stats saved to %s", stats_json)
    return processed_chunks
