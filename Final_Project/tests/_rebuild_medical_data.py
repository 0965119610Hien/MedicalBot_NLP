import ast
import csv
import json
from pathlib import Path

csv_path = Path(r"D:\Final_Project\Final_Project\tests\medical_eval_report.csv")
json_path = Path(r"D:\Final_Project\Final_Project\tests\medical_data.json")

rows = []
with csv_path.open("r", encoding="utf-8") as f:
    reader = csv.DictReader(f)
    for row in reader:
        question = (row.get("user_input") or "").strip()
        response = (row.get("response") or "").strip()
        reference = (row.get("reference") or "").strip()
        contexts_raw = row.get("retrieved_contexts") or "[]"
        try:
            contexts = ast.literal_eval(contexts_raw)
            if not isinstance(contexts, list):
                contexts = [str(contexts)]
        except Exception:
            contexts = [contexts_raw]
        if not question:
            continue
        rows.append({
            "question": question,
            "contexts": contexts,
            "answer": response,
            "ground_truth": reference,
        })

json_path.write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")
print(f"Wrote {len(rows)} samples to {json_path}")
