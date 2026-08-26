# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Building a data iterator must not draw from the default CPU generator.

See the comment in data_samplers.build_pretraining_data_loader; the draw happens on ITERATOR
creation, not on DataLoader construction, so this calls iter().
"""

from types import SimpleNamespace

import torch

import megatron.training.datasets.data_samplers as data_samplers
from tests.unit_tests.test_utilities import Utils


class _Dataset(torch.utils.data.Dataset):
    def __len__(self):
        return 64

    def __getitem__(self, idx):
        return torch.tensor([idx])


class TestDataLoaderRNG:
    def teardown_method(self, method):
        Utils.destroy_model_parallel()

    def test_building_the_iterator_leaves_the_cpu_rng_untouched(self, monkeypatch):
        Utils.initialize_model_parallel(1, 1)
        monkeypatch.setattr(
            data_samplers,
            'get_args',
            lambda: SimpleNamespace(
                dataloader_type='single',
                micro_batch_size=1,
                global_batch_size=Utils.world_size,
                num_workers=0,
                hybrid_context_parallel=False,
            ),
        )
        torch.manual_seed(1234)

        loader = data_samplers.build_pretraining_data_loader(_Dataset(), consumed_samples=0)
        assert loader.generator is not None, "loader has no generator of its own"
        # Tracks _set_random_seed's per-rank seed rather than a fixed or rank-independent value.
        assert loader.generator.initial_seed() == 1234

        before = torch.get_rng_state().clone()
        iter(loader)  # this is where _base_seed is drawn
        assert torch.equal(before, torch.get_rng_state()), (
            "creating the data iterator consumed the default CPU generator; a resumed run does "
            "this AFTER restoring the saved RNG state, so it diverges from the run that saved"
        )
