#!/usr/bin/env python3
"""Select a deterministic, answerable slice of TriviaQA validation."""
import argparse, json, random, re
from pathlib import Path
from datasets import load_dataset

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=240)
    ap.add_argument("--seed", type=int, default=314159)
    ap.add_argument("--out", default="data/triviaqa_selected.jsonl")
    args = ap.parse_args()
    # The no-context configuration avoids downloading unused evidence documents.
    ds = load_dataset("mandarjoshi/trivia_qa", "unfiltered.nocontext", split="validation")
    candidates = []
    for i, row in enumerate(ds):
        q = row["question"].strip()
        aliases = list(dict.fromkeys([row["answer"]["value"]] + row["answer"]["aliases"]))
        # Avoid unusually long/list answers and questions whose grading is inherently ambiguous.
        if 4 <= len(q) <= 220 and aliases and len(row["answer"]["value"]) <= 60:
            candidates.append({"id": row.get("question_id", str(i)), "question": q,
                               "answer": row["answer"]["value"], "aliases": aliases})
    rng = random.Random(args.seed)
    rng.shuffle(candidates)
    selected = candidates[:args.n]
    out = Path(args.out); out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w") as f:
        for x in selected: f.write(json.dumps(x, ensure_ascii=False) + "\n")
    print(json.dumps({"dataset_rows": len(ds), "eligible": len(candidates), "selected": len(selected),
                      "seed": args.seed, "output": str(out)}, indent=2))
if __name__ == "__main__": main()
