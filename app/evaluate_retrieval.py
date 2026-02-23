import json
import math
from collections import defaultdict
from typing import List, Dict, Tuple

from .config import settings
from .rag import load_chunks, load_index, retrieve

def read_jsonl(path: str) -> List[Dict]:
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows

def precision_at_k(retrieved: List[str], relevant: set, k: int) -> float:
    topk = retrieved[:k]
    if k == 0:
        return 0.0
    hits = sum(1 for u in topk if u in relevant)
    return hits / k

def recall_at_k(retrieved: List[str], relevant: set, k: int) -> float:
    if not relevant:
        return 0.0
    topk = retrieved[:k]
    hits = sum(1 for u in topk if u in relevant)
    return hits / len(relevant)

def hitrate_at_k(retrieved: List[str], relevant: set, k: int) -> float:
    topk = retrieved[:k]
    return 1.0 if any(u in relevant for u in topk) else 0.0

def mrr_at_k(retrieved: List[str], relevant: set, k: int) -> float:
    topk = retrieved[:k]
    for i, u in enumerate(topk, start=1):
        if u in relevant:
            return 1.0 / i
    return 0.0

def ndcg_at_k(retrieved: List[str], relevant: set, k: int) -> float:
    # binary relevance: rel=1 if in relevant, else 0
    def dcg(items: List[str]) -> float:
        score = 0.0
        for i, u in enumerate(items, start=1):
            rel = 1.0 if u in relevant else 0.0
            score += (2**rel - 1) / math.log2(i + 1)
        return score

    topk = retrieved[:k]
    dcg_val = dcg(topk)

    # ideal list: all relevant first (up to k)
    ideal = list(relevant)[:k]
    idcg_val = dcg(ideal)
    if idcg_val == 0:
        return 0.0
    return dcg_val / idcg_val

def majority_section(top_sources: List[Dict]) -> str:
    if not top_sources:
        return "unknown"
    counts = defaultdict(int)
    for s in top_sources:
        counts[s.get("section", "unknown")] += 1
    return max(counts.items(), key=lambda x: x[1])[0]

def evaluate(eval_path: str, ks: List[int] = [1,3,5]) -> None:
    chunks = load_chunks(settings.CHUNKS_PATH)
    index = load_index(settings.FAISS_PATH)

    queries = read_jsonl(eval_path)
    if not queries:
        raise RuntimeError(f"No eval queries found in {eval_path}")

    # Aggregate metrics
    agg = {k: defaultdict(float) for k in ks}
    n = 0

    # Optional section accuracy
    section_correct = 0
    section_total = 0

    per_query_rows = []

    for q in queries:
        qid = q.get("id", f"row{n+1}")
        text = q["query"]
        relevant = set(q.get("relevant_urls", []))
        expected_section = q.get("section")

        sources = retrieve(index, chunks, text, top_k=max(ks))
        retrieved_urls = [s["source_url"] for s in sources]

        row = {"id": qid, "query": text}
        for k in ks:
            row[f"P@{k}"] = precision_at_k(retrieved_urls, relevant, k)
            row[f"R@{k}"] = recall_at_k(retrieved_urls, relevant, k)
            row[f"Hit@{k}"] = hitrate_at_k(retrieved_urls, relevant, k)
            row[f"MRR@{k}"] = mrr_at_k(retrieved_urls, relevant, k)
            row[f"nDCG@{k}"] = ndcg_at_k(retrieved_urls, relevant, k)

            agg[k]["P"] += row[f"P@{k}"]
            agg[k]["R"] += row[f"R@{k}"]
            agg[k]["Hit"] += row[f"Hit@{k}"]
            agg[k]["MRR"] += row[f"MRR@{k}"]
            agg[k]["nDCG"] += row[f"nDCG@{k}"]

        # Section accuracy (optional)
        if expected_section:
            pred_section = majority_section(sources[:3])
            row["section_expected"] = expected_section
            row["section_pred"] = pred_section
            section_total += 1
            if pred_section == expected_section:
                section_correct += 1

        # Add top retrieved URLs for auditability
        row["top_urls"] = retrieved_urls[:5]
        per_query_rows.append(row)
        n += 1

    # Print aggregate results
    print("\n=== Retrieval Evaluation Summary ===")
    print(f"Queries evaluated: {n}")
    for k in ks:
        print(f"\n-- @ {k} --")
        print(f"Precision@{k}: {agg[k]['P']/n:.3f}")
        print(f"Recall@{k}:    {agg[k]['R']/n:.3f}")
        print(f"HitRate@{k}:   {agg[k]['Hit']/n:.3f}")
        print(f"MRR@{k}:       {agg[k]['MRR']/n:.3f}")
        print(f"nDCG@{k}:      {agg[k]['nDCG']/n:.3f}")

    if section_total > 0:
        print(f"\nSection accuracy (majority of top-3): {section_correct/section_total:.3f} ({section_correct}/{section_total})")

    # Save per-query report for your write-up
    out_path = "index/retrieval_eval_report.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(per_query_rows, f, indent=2, ensure_ascii=False)
    print(f"\nSaved per-query report: {out_path}")

if __name__ == "__main__":
    evaluate("data/eval_queries.jsonl", ks=[1,3,5])
