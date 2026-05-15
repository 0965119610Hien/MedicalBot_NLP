import os
from urllib.parse import urlparse

import pandas as pd
import pytest
import requests
from datasets import Dataset
from langchain_community.embeddings import HuggingFaceEmbeddings
import instructor
from openai import OpenAI
from ragas import evaluate
from ragas.llms.base import InstructorLLM, InstructorModelArgs
from ragas.metrics import (
    LLMContextPrecisionWithoutReference,
    answer_relevancy,
    faithfulness,
)
from ragas.run_config import RunConfig


def _get_lm_studio_base_url() -> str:
    raw = os.getenv("LM_STUDIO_URL", "http://localhost:1234/v1")
    parsed = urlparse(raw)
    if not parsed.scheme:
        return "http://localhost:1234/v1"

    if parsed.path.endswith("/v1"):
        return raw.rstrip("/")
    if parsed.path.endswith("/v1/chat/completions"):
        return raw[: -len("/v1/chat/completions")]
    if parsed.path == "":
        return f"{raw.rstrip('/')}/v1"

    return raw


def _require_lm_studio(base_url: str) -> None:
    try:
        response = requests.get(f"{base_url}/models", timeout=5)
        if response.status_code >= 400:
            pytest.skip(f"LM Studio not ready at {base_url}")
    except requests.RequestException:
        pytest.skip(f"LM Studio not reachable at {base_url}")


@pytest.fixture(scope="module")
def ragas_llm():
    base_url = _get_lm_studio_base_url()
    _require_lm_studio(base_url)
    model = os.getenv("LM_STUDIO_MODEL", "qwen2.5-3b-instruct")
    client = OpenAI(base_url=base_url, api_key="lm-studio")
    patched_client = instructor.from_openai(client, mode=instructor.Mode.JSON_SCHEMA)
    return InstructorLLM(
        client=patched_client,
        model=model,
        provider="openai",
        model_args=InstructorModelArgs(),
    )


@pytest.fixture(scope="module")
def ragas_embeddings() -> HuggingFaceEmbeddings:
    model = os.getenv("RAGAS_EMBEDDING_MODEL", "intfloat/multilingual-e5-small")
    return HuggingFaceEmbeddings(model_name=model)


@pytest.fixture(scope="module")
def sample_dataset() -> Dataset:
    data = [
        {
            "question": "What are common symptoms of diabetes?",
            "contexts": [
                "Diabetes often causes excessive thirst, frequent urination, unexplained weight loss, fatigue, and blurred vision. These are common symptoms of diabetes.",
            ],
            "answer": "Common symptoms include excessive thirst, frequent urination, unexplained weight loss, fatigue, and blurred vision.",
        },
        {
            "question": "How should adults take aspirin for a mild headache?",
            "contexts": [
                "Aspirin is a pain reliever and fever reducer. Adults typically take 500 mg per dose every 6 to 8 hours if needed, take it after food to reduce stomach irritation, and do not exceed the recommended dose.",
            ],
            "answer": "Adults can take aspirin 500 mg per dose every 6 to 8 hours if needed, and should take it after food. Do not exceed the recommended dose.",
        },
    ]
    return Dataset.from_list(data)


def _assert_scores(scores: pd.DataFrame) -> None:
    for metric in ["faithfulness", "answer_relevancy", "context_precision"]:
        assert (scores[metric] >= 0.8).all(), f"{metric} score below 0.8"


def test_rag_quality_metrics(
    sample_dataset: Dataset, ragas_llm, ragas_embeddings: HuggingFaceEmbeddings
) -> None:
    context_precision = LLMContextPrecisionWithoutReference(name="context_precision")
    results = evaluate(
        sample_dataset,
        metrics=[faithfulness, answer_relevancy, context_precision],
        llm=ragas_llm,
        embeddings=ragas_embeddings,
        run_config=RunConfig(timeout=120, max_retries=2),
    )
    scores = results.to_pandas()
    _assert_scores(scores)
