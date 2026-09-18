import time
import numpy as np
import hisss

print("CUDA:", hisss.cuda_available(), hisss.cuda_last_error())

cfg = hisss.duel_config()

t0 = time.perf_counter()
res = hisss.run_random_rollouts_cuda(
    cfg,
    num_games=1048576,
    max_turns=5000,
    seed=123,
)
dt = time.perf_counter() - t0

print("Dauer:", dt)
print("Spiele:", len(res.turns_played))
print("Terminal:", int(res.terminal.sum()))
print("Mittlere Turns:", float(res.turns_played.mean()))

winner, counts = np.unique(res.winner, return_counts=True)
print("Winner:", dict(zip(winner.tolist(), counts.tolist())))