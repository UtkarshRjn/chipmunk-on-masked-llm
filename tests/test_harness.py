import json

from mdm_chipmunk.eval.harness import run
from mdm_chipmunk.eval.tasks.base import Task, TaskSample
from mdm_chipmunk.methods.dense import DenseMDM
from mdm_chipmunk.models.synthetic import SyntheticMDM


class _ToyTask(Task):
    name = "toy"
    gen_length = 8

    def iter_samples(self, num_samples=None):
        n = num_samples or 3
        for i in range(n):
            yield TaskSample(sample_idx=i, prompt=f"q {i}", gold=str(i), metadata={})

    def score(self, prediction_text, gold):
        return False


def test_harness_writes_jsonl(tmp_path):
    model = SyntheticMDM(seq_len=12, vocab_size=32, num_layers=1, d_model=16, n_heads=2)
    model.eval()
    method = DenseMDM()
    task = _ToyTask()

    out = tmp_path / "run.jsonl"
    summary = run(
        method=method, model=model, task=task,
        num_samples=2, num_steps=2, seed=0, out_path=out,
    )
    assert out.exists()
    rows = [json.loads(line) for line in out.read_text().splitlines()]
    assert len(rows) == 2
    for row in rows:
        assert row["method"] == "dense"
        assert row["task"] == "toy"
        assert row["gen_length"] == 8
        assert row["correct"] is False
        assert row["latency_s"] >= 0
    assert summary.accuracy == 0.0
    assert summary.num_samples == 2
