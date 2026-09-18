from __future__ import annotations

import os
import sys
from pathlib import Path

from ppo4 import PPOAgent4, _DEFAULT_MODEL


DEFAULT_SEARCH_HORIZON = 18
DEFAULT_SEARCH_NODE_BUDGET = 12_000
DEFAULT_PPO5_MODEL_PATH = Path(__file__).with_name(
    "ppo_bs_lstm_cuda_v26_ppo4_best_response_selected.zip"
)


class PPOAgent5(PPOAgent4):
    """PPO4 with a deeper, proof-only self-trap filter.

    Strategy, recurrent state, Blackout encoding, opponent handling, and every
    hard safety invariant are inherited unchanged from PPO4. The only policy
    improvement is a longer exact own-body survival proof (18 rather than 10
    turns) with a bounded 12,000-state budget. If the budget is exhausted, the
    result is UNKNOWN and cannot veto PPO's action.

    This is intentionally not MCTS: it has no opponent simulation, sampling,
    rollout evaluation, tree policy, visit counts, or time-dependent result.
    """

    def __init__(
        self,
        model_path: str | os.PathLike[str] | None | object = _DEFAULT_MODEL,
        *,
        symmetries: int | None = None,
        safety_search: bool | None = None,
        search_horizon: int | None = None,
        search_node_budget: int | None = None,
        device: str | None = None,
    ):
        if model_path is _DEFAULT_MODEL:
            model_path = os.environ.get(
                "PPO5_MODEL_PATH",
                os.environ.get("PPO_MODEL_PATH", str(DEFAULT_PPO5_MODEL_PATH)),
            )
        if search_horizon is None:
            search_horizon = int(
                os.environ.get(
                    "PPO5_SEARCH_HORIZON",
                    os.environ.get(
                        "PPO_SEARCH_HORIZON", str(DEFAULT_SEARCH_HORIZON)
                    ),
                )
            )
        if search_node_budget is None:
            search_node_budget = int(
                os.environ.get(
                    "PPO5_SEARCH_NODE_BUDGET",
                    os.environ.get(
                        "PPO_SEARCH_NODE_BUDGET", str(DEFAULT_SEARCH_NODE_BUDGET)
                    ),
                )
            )
        super().__init__(
            model_path,
            symmetries=symmetries,
            safety_search=safety_search,
            search_horizon=search_horizon,
            search_node_budget=search_node_budget,
            device=device,
        )
        self._base_code_fingerprint = self._code_fingerprint
        self._code_fingerprint = self._fingerprint(Path(__file__).resolve())

    def get_name(self):
        return "Der Snaketürke v5"

    def get_diagnostics(self) -> dict[str, object]:
        result = super().get_diagnostics()
        result.update(
            {
                "agent": type(self).__name__,
                "inference": "deep-proof-only-self-trap-filter",
                "base_code_sha256": self._base_code_fingerprint,
            }
        )
        return result


if __name__ == "__main__":
    from battlesnake_server import start_server

    if len(sys.argv) < 2:
        print(f"Usage: python {sys.argv[0]} <port>")
        raise SystemExit(1)

    start_server(agent=PPOAgent5(), port=int(sys.argv[1]))
