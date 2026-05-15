from dataclasses import dataclass
from pathlib import Path
import os
from typing import Optional


def _get_int(name: str, default: int) -> int:
    value = os.getenv(name)
    if value is None:
        return default
    try:
        return int(value)
    except ValueError:
        return default


@dataclass(frozen=True)
class Paths:
    root_dir: Path
    clean_data_dir: Path
    embedding_dir: Path
    data_pipeline_dir: Path
    search_index_dir: Path
    processed_chunks: Path
    processed_stats: Path
    embedding_abbrev_map: Path
    clean_abbrev_map: Path


@dataclass(frozen=True)
class InputValidationConfig:
    min_chars: int
    max_chars: int


@dataclass(frozen=True)
class LMStudioConfig:
    url: str
    model: str


@dataclass(frozen=True)
class AppConfig:
    top_k: int


@dataclass(frozen=True)
class PipelineDefaults:
    chunk_size: int
    chunk_overlap: int
    min_words: int


@dataclass(frozen=True)
class ProjectConfig:
    paths: Paths
    validation: InputValidationConfig
    lm_studio: LMStudioConfig
    app: AppConfig
    pipeline: PipelineDefaults


_CONFIG: Optional[ProjectConfig] = None


def get_config() -> ProjectConfig:
    global _CONFIG
    if _CONFIG is not None:
        return _CONFIG

    root_dir = Path(__file__).resolve().parent
    clean_data_dir = root_dir / "CleanData"
    embedding_dir = root_dir / "Embedding"
    data_pipeline_dir = root_dir / "data_pipeline"
    search_index_dir = embedding_dir / "medical_search_index"

    paths = Paths(
        root_dir=root_dir,
        clean_data_dir=clean_data_dir,
        embedding_dir=embedding_dir,
        data_pipeline_dir=data_pipeline_dir,
        search_index_dir=search_index_dir,
        processed_chunks=clean_data_dir / "processed_chunks.json",
        processed_stats=clean_data_dir / "processed_chunks_stats.json",
        embedding_abbrev_map=embedding_dir / "abbreviation_map.txt",
        clean_abbrev_map=clean_data_dir / "abbreviation_map.txt",
    )

    validation = InputValidationConfig(
        min_chars=_get_int("INPUT_MIN_CHARS", 3),
        max_chars=_get_int("INPUT_MAX_CHARS", 500),
    )

    lm_studio = LMStudioConfig(
        url=os.getenv("LM_STUDIO_URL", "http://localhost:1234/v1/chat/completions"),
        model=os.getenv("LM_STUDIO_MODEL", "qwen2.5-3b-instruct"),
    )

    app = AppConfig(
        top_k=_get_int("APP_TOP_K", 3),
    )

    pipeline = PipelineDefaults(
        chunk_size=_get_int("PIPELINE_CHUNK_SIZE", 1500),
        chunk_overlap=_get_int("PIPELINE_CHUNK_OVERLAP", 150),
        min_words=_get_int("PIPELINE_MIN_WORDS", 10),
    )

    _CONFIG = ProjectConfig(
        paths=paths,
        validation=validation,
        lm_studio=lm_studio,
        app=app,
        pipeline=pipeline,
    )
    return _CONFIG
