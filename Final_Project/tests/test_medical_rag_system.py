import os
import json
import pytest
import pandas as pd
import instructor
from openai import OpenAI
from datasets import Dataset
from langchain_huggingface import HuggingFaceEmbeddings

# Ragas components
from ragas import evaluate
from ragas.llms.base import InstructorLLM, InstructorModelArgs
from ragas.metrics import (
    faithfulness,
    answer_relevancy,
    answer_correctness,
    LLMContextPrecisionWithoutReference
)
from ragas.run_config import RunConfig

# --- 1. SETUP INFERENCE (LM STUDIO) ---
@pytest.fixture(scope="module")
def ragas_llm():
    base_url = os.getenv("LM_STUDIO_URL", "http://localhost:1234/v1")
    model_name = os.getenv("LM_STUDIO_MODEL", "qwen2.5-3b-instruct") # Hoặc model bạn đang load
    
    client = OpenAI(base_url=base_url, api_key="lm-studio")
    # Patch client để xử lý JSON output từ model nhỏ ổn định hơn
    patched_client = instructor.from_openai(client, mode=instructor.Mode.JSON_SCHEMA)
    
    return InstructorLLM(
        client=patched_client,
        model=model_name,
        provider="openai",
        model_args=InstructorModelArgs(),
    )

@pytest.fixture(scope="module")
def ragas_embeddings():
    # Model embedding đa ngôn ngữ tốt nhất cho tiếng Việt/Anh hiện tại
    return HuggingFaceEmbeddings(model_name="intfloat/multilingual-e5-small")

# --- 2. LOAD DATA FROM JSON ---
@pytest.fixture(scope="module")
def medical_dataset():
    # Đọc các test cases từ file JSON cùng thư mục
    data_path = os.path.join(os.path.dirname(__file__), "medical_data.json")
    with open(data_path, "r", encoding="utf-8-sig") as f:
        data = json.load(f)
    return Dataset.from_list(data)

# --- 3. EXECUTE EVALUATION ---
def test_rag_quality_and_safety(medical_dataset, ragas_llm, ragas_embeddings):
    # Khởi tạo metric đánh giá truy xuất của Hybrid RAG
    context_precision = LLMContextPrecisionWithoutReference(name="context_precision")
    
    print(f"\nĐang đánh giá {len(medical_dataset)} test cases...")
    
    results = evaluate(
        medical_dataset,
        metrics=[
            faithfulness, 
            answer_relevancy, 
            answer_correctness,
            context_precision
        ],
        llm=ragas_llm,
        embeddings=ragas_embeddings,
        run_config=RunConfig(timeout=180, max_retries=3)
    )

    # Xuất báo cáo
    df = results.to_pandas()
    report_dir = os.path.dirname(__file__)
    df.to_csv(
        os.path.join(report_dir, "medical_eval_report.csv"),
        index=False,
        encoding="utf-8-sig",
    )
    
    # Assertions
    avg_scores = df[['faithfulness', 'answer_relevancy', 'answer_correctness']].mean()
    print("\n--- KẾT QUẢ TRUNG BÌNH ---")
    print(avg_scores)

    # Lưu kết quả dạng txt để dễ đọc nhanh
    txt_path = os.path.join(report_dir, "medical_eval_report.txt")
    with open(txt_path, "w", encoding="utf-8") as f:
        f.write("MEDICAL RAG EVAL REPORT\n")
        f.write(f"Total cases: {len(medical_dataset)}\n")
        f.write("Average scores:\n")
        f.write(f"  faithfulness: {avg_scores['faithfulness']:.4f}\n")
        f.write(f"  answer_relevancy: {avg_scores['answer_relevancy']:.4f}\n")
        f.write(f"  answer_correctness: {avg_scores['answer_correctness']:.4f}\n")
        if "context_precision" in df.columns:
            f.write(f"  context_precision: {df['context_precision'].mean():.4f}\n")

    # Y tế cần độ chính xác cực cao (0.8 là ngưỡng tối thiểu)
    threshold = 0.8
    assert avg_scores['faithfulness'] >= threshold, "Cảnh báo: Tỉ lệ ảo giác vượt mức cho phép!"
    assert avg_scores['answer_correctness'] >= threshold, "Cảnh báo: Độ chính xác y khoa không đạt chuẩn!"

if __name__ == "__main__":
    # Để chạy trực tiếp không cần lệnh pytest
    pytest.main([__file__])