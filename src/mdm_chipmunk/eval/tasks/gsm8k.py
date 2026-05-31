from __future__ import annotations

import re
from typing import Any, Iterator

from .base import Task, TaskSample

_ANSWER_REGEX = re.compile(r"#### *(-?\d[\d,]*(?:\.\d+)?)")
_LAST_NUMBER_REGEX = re.compile(r"(-?\d[\d,]*(?:\.\d+)?)(?!.*\d)", re.DOTALL)

_FEW_SHOT_EXAMPLES = [
    (
        "Natalia sold clips to 48 of her friends in April, and then she sold half as many "
        "clips in May. How many clips did Natalia sell altogether in April and May?",
        "Natalia sold 48/2 = 24 clips in May. In total she sold 48+24 = 72 clips. #### 72",
    ),
    (
        "Weng earns $12 an hour for babysitting. Yesterday, she just did 50 minutes of "
        "babysitting. How much did she earn?",
        "50 minutes is 50/60 = 5/6 of an hour. She earned 12 * 5/6 = $10. #### 10",
    ),
    (
        "Betty is saving money for a new wallet which costs $100. Betty has only half of "
        "the money she needs. Her parents decided to give her $15 for that purpose, and "
        "her grandparents twice as much as her parents. How much more money does Betty "
        "need to buy the wallet?",
        "Half of $100 is $50. Grandparents give 2*$15 = $30. Total = 50+15+30 = $95. "
        "She needs 100-95 = $5 more. #### 5",
    ),
    (
        "James writes a 3-page letter to 2 different friends twice a week. How many pages "
        "does he write a year?",
        "Each week he writes 3*2*2 = 12 pages. Per year: 12*52 = 624 pages. #### 624",
    ),
    (
        "Mark has a garden with flowers. He planted plants of three different colors. Ten "
        "of them are yellow, and there are 80% more of those in purple. There are only "
        "25% as many green flowers as there are yellow and purple flowers. How many "
        "flowers does Mark have in his garden?",
        "Purple = 10 + 10*0.8 = 18. Yellow + purple = 28. Green = 28*0.25 = 7. "
        "Total = 10+18+7 = 35. #### 35",
    ),
]


class GSM8K(Task):
    name = "gsm8k"

    def __init__(
        self,
        num_shots: int = 5,
        gen_length: int = 256,
        split: str = "test",
        hf_id: str = "openai/gsm8k",
        subset: str = "main",
        **cfg: Any,
    ):
        super().__init__(**cfg)
        self.num_shots = num_shots
        self.gen_length = gen_length
        self.split = split
        self.hf_id = hf_id
        self.subset = subset
        self._dataset = None

    def _load_dataset(self) -> Any:
        if self._dataset is None:
            from datasets import load_dataset

            self._dataset = load_dataset(self.hf_id, self.subset, split=self.split)
        return self._dataset

    def _build_prompt(self, question: str) -> str:
        shots = _FEW_SHOT_EXAMPLES[: self.num_shots]
        parts = []
        for q, a in shots:
            parts.append(f"Question: {q}\nAnswer: {a}\n")
        parts.append(f"Question: {question}\nAnswer:")
        return "\n".join(parts)

    @staticmethod
    def _parse_gold(answer_field: str) -> str:
        m = _ANSWER_REGEX.search(answer_field)
        return (m.group(1) if m else answer_field.strip()).replace(",", "")

    @staticmethod
    def _parse_prediction(text: str) -> str | None:
        m = _ANSWER_REGEX.search(text)
        if m:
            return m.group(1).replace(",", "")
        m = _LAST_NUMBER_REGEX.search(text)
        return m.group(1).replace(",", "") if m else None

    def iter_samples(self, num_samples: int | None = None) -> Iterator[TaskSample]:
        ds = self._load_dataset()
        n = len(ds) if num_samples is None else min(num_samples, len(ds))
        for i in range(n):
            row = ds[i]
            yield TaskSample(
                sample_idx=i,
                prompt=self._build_prompt(row["question"]),
                gold=self._parse_gold(row["answer"]),
                metadata={"question": row["question"], "answer_full": row["answer"]},
            )

    def score(self, prediction_text: str, gold: Any) -> bool:
        pred = self._parse_prediction(prediction_text)
        if pred is None:
            return False
        try:
            return float(pred) == float(gold)
        except ValueError:
            return pred == gold
