from __future__ import annotations

import argparse
import os

import torch

import hisss
from train_cuda import compare_policies_cuda, load_sb3_recurrent_ppo


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare two ppo.py-compatible RecurrentPPO zip models with Battlesnake CUDA."
    )
    parser.add_argument("model_a", help="First PPO zip path")
    parser.add_argument("model_b", help="Second PPO zip path")
    parser.add_argument("--games", type=int, default=512)
    parser.add_argument("--num-envs", type=int, default=256)
    parser.add_argument("--max-turns", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument(
        "--layout",
        choices=("copies", "duel"),
        default="copies",
        help="copies: seats contain A,B,A,B; duel: one A, one B, two filler snakes",
    )
    parser.add_argument(
        "--fill",
        choices=("random", "hungry"),
        default="random",
        help="Filler policy for --layout duel",
    )
    parser.add_argument(
        "--stochastic",
        action="store_true",
        help="Sample from masked logits instead of greedy masked actions",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("PyTorch CUDA is not available")
    if not hisss.cuda_available():
        raise RuntimeError(hisss.cuda_last_error())

    device = torch.device("cuda")
    model_a = load_sb3_recurrent_ppo(args.model_a, device=device)
    model_b = load_sb3_recurrent_ppo(args.model_b, device=device)
    stats = compare_policies_cuda(
        model_a,
        model_b,
        games=args.games,
        num_envs=args.num_envs,
        max_turns=args.max_turns,
        seed=args.seed,
        layout=args.layout,
        fill=args.fill,
        deterministic=not args.stochastic,
        device=device,
    )

    name_a = os.path.basename(args.model_a)
    name_b = os.path.basename(args.model_b)
    print("=== CUDA PPO comparison ===")
    print(f"Games: {int(stats['games'])}")
    print(f"Layout: {args.layout}  Fill: {args.fill}")
    print(f"Average turns: {stats['avg_turns']:.1f}")
    print(f"{name_a}: wins={int(stats['a_wins'])} win_rate={stats['a_win_rate']:.3f}")
    print(f"{name_b}: wins={int(stats['b_wins'])} win_rate={stats['b_win_rate']:.3f}")
    if args.layout == "duel":
        print(f"Fillers: wins={int(stats['fill_wins'])} win_rate={stats['fill_win_rate']:.3f}")
    print(f"Draws: {int(stats['draws'])} draw_rate={stats['draw_rate']:.3f}")
    print(f"Deaths: {name_a}={int(stats['a_deaths'])} {name_b}={int(stats['b_deaths'])}")


if __name__ == "__main__":
    main()
