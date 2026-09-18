from __future__ import annotations

import pytest

from ppo5 import (
    DEFAULT_PPO5_MODEL_PATH,
    DEFAULT_SEARCH_HORIZON,
    DEFAULT_SEARCH_NODE_BUDGET,
    PPOAgent5,
)


def test_ppo5_defaults_extend_ppo4_proof_depth():
    assert DEFAULT_SEARCH_HORIZON == 18
    assert DEFAULT_SEARCH_NODE_BUDGET == 12_000


@pytest.mark.model
@pytest.mark.skipif(
    not DEFAULT_PPO5_MODEL_PATH.is_file(),
    reason="Optional PPO5 deployment weights are not included in the source release",
)
def test_ppo5_deployment_model_loads():
    agent = PPOAgent5(model_path=DEFAULT_PPO5_MODEL_PATH, device="cpu")
    assert agent.model_path == DEFAULT_PPO5_MODEL_PATH.resolve()


def test_ppo5_has_distinct_deployment_identity():
    agent = PPOAgent5.__new__(PPOAgent5)
    assert agent.get_name() == "Der Snaketürke v5"
