import json
import logging
from dataclasses import asdict
from pathlib import Path
from typing import List

logger = logging.getLogger(__name__)

METHODS = {"direct_scoring", "rubric", "anchor", "pairwise", "tournament"}


class ResultStore:
    def __init__(self, store_dir: str):
        self.store_dir = Path(store_dir)
        self.store_dir.mkdir(parents=True, exist_ok=True)

    def save(self, exam_id: str, method: str, results: List) -> Path:
        path = self.store_dir / f"{exam_id}_{method}_results.json"
        serialized = []
        for r in results:
            if hasattr(r, "__dataclass_fields__"):
                serialized.append(asdict(r))
            elif isinstance(r, dict):
                serialized.append(r)
            else:
                serialized.append(str(r))

        data = {
            "exam_id": exam_id,
            "method": method,
            "count": len(serialized),
            "results": serialized,
        }
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        logger.info(f"Saved {len(results)} {method} results to {path}")
        return path

    def load(self, exam_id: str, method: str) -> List[dict]:
        path = self.store_dir / f"{exam_id}_{method}_results.json"
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return data["results"]

    def list_results(self, exam_id: str) -> List[str]:
        return [
            p.stem.split("_", maxsplit=len(exam_id.split("_")))[-1].replace("_results", "")
            for p in self.store_dir.glob(f"{exam_id}_*_results.json")
        ]
