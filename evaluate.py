import csv
from pathlib import Path

from rag_engine import MAX_COSINE_DISTANCE, get_index_count, retrieve

ROOT = Path(__file__).resolve().parent
TEST_FILE = ROOT / "test_questions.csv"


def main() -> None:
    if get_index_count() == 0:
        print("No vector index found. Run: python ingest.py")
        return

    rows = list(csv.DictReader(TEST_FILE.open(encoding="utf-8-sig", newline="")))
    supported_total = 0
    supported_hits = 0
    unsupported_total = 0
    unsupported_abstentions = 0

    print(f"\nRetrieval evaluation (top_k=4, cosine-distance cutoff={MAX_COSINE_DISTANCE:.2f})")
    print("-" * 100)

    for row in rows:
        question = row["question"]
        expected = row["expected_evidence"].strip()
        hits = retrieve(question, top_k=4)
        sources = [hit["source"] for hit in hits]
        best_distance = min((hit["distance"] for hit in hits), default=999.0)
        passes_cutoff = best_distance <= MAX_COSINE_DISTANCE

        if expected.lower() == "none":
            unsupported_total += 1
            abstained = not passes_cutoff
            unsupported_abstentions += int(abstained)
            status = "ABSTAIN" if abstained else "REVIEW: POSSIBLE FALSE MATCH"
        else:
            supported_total += 1
            hit = any(source.endswith(expected) for source in sources) and passes_cutoff
            supported_hits += int(hit)
            status = "HIT" if hit else "MISS"

        print(f"{status:26} | best distance={best_distance:.3f} | {question}")
        print(f"{'':26} | expected={expected} | retrieved={', '.join(sources[:4])}")

    print("-" * 100)
    if supported_total:
        print(f"Source Hit@4: {supported_hits}/{supported_total} = {supported_hits/supported_total:.1%}")
    if unsupported_total:
        print(
            "Unsupported-query abstention: "
            f"{unsupported_abstentions}/{unsupported_total} = "
            f"{unsupported_abstentions/unsupported_total:.1%}"
        )
    print("\nThese are retrieval-side checks only. Review answer faithfulness and correctness manually.")
    print("Tune MAX_COSINE_DISTANCE in .env using these observed distances; do not treat the default as universal.")


if __name__ == "__main__":
    main()
