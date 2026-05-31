import torch

from mdm_chipmunk.methods.dense import DenseMDM, _linear_unmask_schedule
from mdm_chipmunk.models.synthetic import SyntheticMDM


def test_linear_unmask_schedule_sums_to_total():
    sched = _linear_unmask_schedule(10, 4)
    assert sum(sched) == 10
    assert len(sched) == 4
    assert sched == [3, 3, 2, 2]


def test_linear_unmask_schedule_zero_steps():
    assert _linear_unmask_schedule(10, 0) == []


def test_dense_generates_and_terminates_with_no_masks():
    model = SyntheticMDM(seq_len=16, vocab_size=32, num_layers=1, d_model=16, n_heads=2)
    model.eval()
    prompt_ids = torch.tensor([[5, 6, 7, 8]], dtype=torch.long)
    method = DenseMDM()
    result = method.generate(
        model=model, prompt_ids=prompt_ids, gen_length=8, num_steps=4, seed=0
    )
    assert result.output_ids.shape == (1, 8)
    # All masked positions in the generation region should be filled.
    assert (result.output_ids != model.mask_token_id).all()
    assert result.telemetry.total_time_s > 0
    assert len(result.telemetry.step_times_s) <= 4
