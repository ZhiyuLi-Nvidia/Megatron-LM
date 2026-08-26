# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Does every rank get its OWN cuda_rng_tracker state back from a checkpoint?

The tracker streams are seeded per rank (tensor_parallel/random.py), e.g.
``expert_parallel_seed = seed + 1024 + 100 * ep_rank + etp_rank``. When ``get_rng_state`` keyed on
``(pp, tp)`` with ``replica_id=dp_cp_rank``, expert parallelism became a replica axis: one rank's
state was written and its peers restored it, so only expert-parallel rank 0 resumed on the stream
it had saved.
"""

from types import SimpleNamespace

import pytest
import torch

import megatron.training.checkpointing as checkpointing
from megatron.core import mpu, tensor_parallel
from megatron.core.dist_checkpointing import load, save
from tests.unit_tests.dist_checkpointing import TempNamedDir
from tests.unit_tests.test_utilities import Utils

# (tp, pp, cp, ep), each satisfying tp * pp * cp * dp == world_size with ep dividing dp: the
# minimal expert-parallel repro, then ep crossed with tp and with cp. cp earns its case because a
# key of (pp, tp, dp) rather than (pp, tp, dp_cp) passes at cp=1 and collides differing ranks at
# cp=2.
LAYOUTS = [(1, 1, 1, 2), (2, 1, 1, 2), (1, 1, 2, 2)]


@pytest.mark.parametrize(('tp', 'pp', 'cp', 'ep'), LAYOUTS)
class TestRNGStateCheckpoint:
    def teardown_method(self, method):
        Utils.destroy_model_parallel()

    def test_each_rank_restores_the_state_it_saved(
        self, tmp_path_dist_ckpt, monkeypatch, tp, pp, cp, ep
    ):
        Utils.initialize_model_parallel(
            tp, pp, context_parallel_size=cp, expert_model_parallel_size=ep
        )
        # get_rng_state reads only args.data_parallel_random_init.
        monkeypatch.setattr(
            checkpointing, 'get_args', lambda: SimpleNamespace(data_parallel_random_init=False)
        )
        # Seeds every tracker stream as a function of this rank's position on each parallel axis.
        tensor_parallel.model_parallel_cuda_manual_seed(123)

        def rng_sharded_object():
            return checkpointing.get_rng_state(
                'torch_dist',
                mpu.get_tensor_model_parallel_group(),
                mpu.get_pipeline_model_parallel_group(),
                dp_cp_group=mpu.get_data_parallel_group(with_context_parallel=True),
                dp_group=mpu.get_data_parallel_group(),
            )

        def tracker_states():
            # get_states() copies the dict but not the tensors, so clone to survive re-seeding.
            return {
                k: v.clone() for k, v in tensor_parallel.get_cuda_rng_tracker().get_states().items()
            }

        expected = tracker_states()
        assert expected, "no tracker states to compare -- the test would prove nothing"

        with TempNamedDir(tmp_path_dist_ckpt / f'rng_{tp}_{pp}_{cp}_{ep}') as ckpt_dir:
            # save() validates access integrity by default, so a key that does not tile the grid
            # fails here rather than silently dropping a shard.
            save({'rng_state': rng_sharded_object()}, ckpt_dir)

            # Perturb every stream, so a load that silently restores nothing cannot pass.
            tensor_parallel.model_parallel_cuda_manual_seed(456)
            assert any(
                not torch.equal(expected[k], v) for k, v in tracker_states().items()
            ), "re-seeding did not change the tracker -- the assertion below would be vacuous"

            loaded = load({'rng_state': rng_sharded_object()}, ckpt_dir)['rng_state']

        restored = loaded[0]['rng_tracker_states']
        for name, want in expected.items():
            assert name in restored, f"'{name}' missing after load on rank {Utils.rank}"
            assert torch.equal(want, restored[name]), (
                f"rank {Utils.rank} restored a different '{name}' stream than it saved "
                f"(tp={tp} pp={pp} cp={cp} ep={ep}) -- the shard key does not separate this rank "
                f"from its peers on some parallel axis"
            )
