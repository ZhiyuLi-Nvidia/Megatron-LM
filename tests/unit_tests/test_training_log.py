# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

import re
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch

from megatron.training import training
from tests.unit_tests.test_utilities import Utils


class _IntervalTimer:
    def __init__(self):
        self.seconds = 0.0
        self.resets = []

    def elapsed(self, *, barrier, reset):
        elapsed = self.seconds
        self.resets.append(reset)
        if reset:
            self.seconds = 0.0
        return elapsed


class TestTrainingLog:
    def setup_method(self):
        Utils.initialize_model_parallel(1, 1)

    def teardown_method(self):
        Utils.destroy_model_parallel()

    @pytest.mark.parametrize(
        "first_iteration,log_interval,expected,resets",
        [
            (1, 1, [(i, float(i)) for i in range(1, 7)], [True] * 6),
            (5, 1, [(i + 4, float(i)) for i in range(1, 7)], [True] * 6),
            (5, 5, [(5, 1.0), (10, 4.0)], [True, True]),
            (2, 5, [(2, 1.0), (5, 2.5)], [False, True]),
        ],
    )
    def test_first_iteration_logging(
        self, monkeypatch, first_iteration, log_interval, expected, resets
    ):
        args = SimpleNamespace(
            timing_log_level=0,
            micro_batch_size=1,
            data_parallel_size=1,
            gtp_weight_remat_size=1,
            world_size=1,
            seq_length=8,
            log_interval=log_interval,
            train_iters=20,
            consumed_train_samples=0,
            skipped_train_samples=0,
            perform_rl_step=False,
            num_experts=None,
            mtp_num_layers=None,
            dsa_indexer_loss_coeff=None,
            profile_ranks=[],
            record_memory_history=False,
            log_timers_to_tensorboard=False,
            log_throughput=False,
            log_energy=False,
            log_memory_interval=None,
            rl_profile=False,
        )
        timer = _IntervalTimer()
        timers = Mock(return_value=timer)
        lines = []
        monkeypatch.setattr(training, "get_args", lambda: args)
        monkeypatch.setattr(training, "get_timers", lambda: timers)
        for getter in (
            "get_tensorboard_writer",
            "get_wandb_writer",
            "get_one_logger",
            "get_energy_monitor",
            "get_telemetry",
        ):
            monkeypatch.setattr(training, getter, lambda: None)
        monkeypatch.setattr(training, "get_num_microbatches", lambda: 1)
        monkeypatch.setattr(training, "has_rl_utils", False)
        monkeypatch.setattr(training, "print_rank_last", lines.append)
        monkeypatch.setattr(training, "num_floating_point_operations", lambda *a, **kw: 1)
        monkeypatch.setattr(
            training, "reduce_max_stat_across_model_parallel_group", lambda value, **kw: value
        )
        monkeypatch.setattr(training.one_logger_utils, "track_app_tag", Mock())
        monkeypatch.setattr(training.one_logger_utils, "track_e2e_metrics", Mock())

        total_loss_dict = {}
        for offset in range(6):
            iteration = first_iteration + offset
            args.consumed_train_samples = iteration
            timer.seconds += offset + 1
            training.training_log(
                loss_dict={"lm loss": torch.tensor([float(offset + 1)], device="cuda")},
                total_loss_dict=total_loss_dict,
                learning_rate=1e-4,
                iteration=iteration,
                loss_scale=1.0,
                report_memory_flag=False,
                skipped_iter=0,
                grad_norm=None,
                params_norm=None,
                num_zeros_in_grad=None,
                max_attention_logit=None,
                is_first_iteration=offset == 0,
            )

        actual = [
            (
                int(re.search(r"iteration\s+(\d+)", line)[1]),
                float(re.search(r"lm loss: ([\d.E+-]+)", line)[1]),
                float(re.search(r"iteration \(ms\): ([\d.]+)", line)[1]),
            )
            for line in lines
        ]
        assert actual == [(iteration, loss, loss * 1000) for iteration, loss in expected]
        assert timer.resets == resets
