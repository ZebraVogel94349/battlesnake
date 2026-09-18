# Der Snaketürke - Battlesnake Blackout (7th)

This write-up describes our entry, **Der Snaketürke**, in the [Battlesnake Blackout competition](https://www.tnt.uni-hannover.de/bs-blackout-2026/). Our snake placed seventh out of 66 entries in the first round and seventh out of eight finalists in the second round.

Blackout's restricted visibility makes remembering previously observed information particularly useful. Our approach combined a recurrent policy trained with Proximal Policy Optimization (PPO), a CUDA game simulator, a diverse opponent pool, and additional safety checks during inference.

## Model Architecture

Our model uses an actor–critic architecture trained with PPO. The actor receives a 29 × 29 observation centered on the snake’s head. Its nine channels encode food, board boundaries, the snake’s own body, head, health and tail, visible enemy bodies and heads, and the visibility mask.

The actor first processes this observation through three convolutional layers with 32, 64, and 64 filters, each using a 3 × 3 kernel and ReLU activation. The second and third layers downsample the feature maps with a stride of 2. The resulting feature maps are projected into a 256-dimensional vector, which is passed to an LSTM.

The LSTM carries a 256-dimensional hidden state and a 256-dimensional cell state across turns, allowing the model to retain information about food and opponents that are no longer visible. Its output passes through a policy head with two hidden layers of 128 units each, both using Tanh activations. A final linear layer produces four action logits in the order **up, down, left, right**. During inference, the highest-scoring action is selected subject to the safety checks described below.

The critic uses a separate CNN and LSTM with the same feature and hidden-state dimensions. Its value head also has two 128-unit hidden layers with Tanh activations, followed by a linear layer that outputs a single estimate of the remaining return. In addition to the actor’s nine channels, the critic receives five privileged channels containing information from the full game state without fog. These provide additional information for value estimation during training. The critic is not used during inference.

## Training

We trained the model using a CUDA reimplementation of the Hisss simulator. For training, CUDA kernels perform the simulation, with Python bindings connecting the native backend to PyTorch. This allowed us to simulate approximately 20,000–40,000 game steps per second on an NVIDIA RTX 3070.

We tried several hyperparameter configurations, repeatedly continuing training from earlier checkpoints. **V25 was the model version used in the final tournament; V26 was not used.** The V25 training configuration supported 1,024–2,048 parallel games, with 128 steps per environment before each PPO update and three PPO epochs per rollout. We used γ = 1, a generalized advantage estimation (GAE) parameter λ = 0.97, a cosine learning-rate schedule, and a value-loss coefficient of 0.5. Further details appear in [Hyperparameters](#hyperparameters).

The V25 configuration retained up to 32 historical policy checkpoints, with up to 13 active at a time. Opponents also included heuristic agents with different strategies: random movement, food seeking, aggressive pursuit, and avoidance of other snakes.

Opponent selection combined prioritized self-play with an approximate Nash-based weighting scheme. In the V25 configuration, evaluations against the opponent pool were scheduled every 250 PPO updates. We maintained a history of evaluation scores, with rows representing successive learner evaluation points and columns representing opponents. An approximate zero-sum solver used this matrix to derive a challenging mixture of opponents, which was used to compute weights for the next training interval. Opponents with very small weights were omitted from the active training mix because evaluating many different opponent models in parallel substantially reduced throughput.

Our goal was to develop a snake that could compete against a wide range of strategies. Before introducing the diverse opponent pool, we trained only against earlier versions of our own model. This led to overfitting: the model performed well against its predecessors but became worse against simple heuristic agents.

The final model accumulated approximately 15 billion game steps of training. We stopped when its mean win rate against the opponent pool no longer improved.

## Rewards

The baseline reward follows the tournament scoring system: **2/1/0/0 points** for first through fourth place.

We originally paid all placement points at the end of the game. Under discounting, this made long wins far less attractive than quick second places. We therefore switched to awarding points incrementally, as soon as they were guaranteed: one point upon reaching the last two surviving snakes, and one more for winning. We eventually removed discounting entirely, as discussed in [Hyperparameters](#hyperparameters).

Experiments with additional rewards taught us several lessons. Direct rewards for eating food prevented starvation in our experiments but encouraged excessive growth, which was a disadvantage in the endgame. After removing these rewards, we observed that the model avoided starvation while often staying short, for example by eating only when health was low.

Penalizing invalid moves also produced an unintended incentive. In early training runs, invalid moves were penalized and then replaced with valid ones, allowing the agent to survive and accumulate further penalties. The model learned to end games early instead. We interpreted this as a form of specification gaming and replaced the penalties with action masking. We later distinguished between moves known to be fatal and moves that were merely risky. Action masking was used during both training and inference.

Finally, we added potential-based reward shaping ([Ng, Harada, and Russell, 1999](https://ai.stanford.edu/~ang/papers/shaping-icml99.pdf)). It adds a difference in a bounded potential Φ to the base reward:

```text
r′ = r + γ · Φ(s′) − Φ(s)

Φ(s) = 0.05 · tanh((L − 3) / 6)       # own length L, saturating
     + 0.02 · health                  # own health, normalized to [0, 1]
     + 0.03 · clamp((m − 1) / 3, 0, 1) # m: moves allowed by the observable mask
```

All three terms use information available in the agent’s fogged observation, so the shaping introduces no additional information unavailable during inference. With γ = 1, the shaping terms telescope over a complete episode to Φ(s_T) − Φ(s_0). We set the terminal potential Φ(s_T) to zero, leaving only −Φ(s_0), which does not depend on the actions taken from that starting state.

Under these conditions, shaping preserves the episode-return objective. Growth can provide positive shaping feedback locally without becoming an additional objective in its own right.

## Hyperparameters

After trying discount factors between 0.9 and 0.99, we chose γ = 1. Finite episodes keep returns bounded, although this does not guarantee stable optimization. The following example illustrates the incentive created by our **original reward scheme, which paid all placement points at the end of the game**:

| Result                | Turn | Discounted value       |
|-----------------------|------|------------------------|
| second after 50 turns | 50   | 1 × 0.99⁵⁰ = **0.61**  |
| win after 300 turns   | 300  | 2 × 0.99³⁰⁰ = **0.10** |

With that original scheme and γ = 0.99, a quick second place has roughly six times the discounted value of a win after a long endgame. Incremental payouts reduce this distortion; choosing γ = 1 removes the preference for earlier rewards altogether, aligning the return with undiscounted tournament points.

During late training, we wanted policy updates to be small refinements. We used `target_kl` to stop the remaining PPO optimization for a rollout when a minibatch’s approximate KL divergence exceeded `1.5 × target_kl`. The table below summarizes settings from successive training versions.

| Version | Peak LR | clip ε | target_kl | ent_coef         |
|---------|---------|--------|-----------|------------------|
| V18     | 5e-6    | 0.20   | 0.012     | 0.012 → 0.0025   |
| V20     | 1e-5    | 0.20   | 0.012     | 0.006 → 0.0025   |
| V21     | 5e-5    | 0.15   | 0.008     | 0.003 → 0.0015   |
| V22     | 2.5e-5  | 0.15   | 0.008     | 0.0035 → 0.0015  |
| V23     | 1.5e-5  | 0.12   | 0.006     | 0.0020 → 0.0008  |
| V24     | 3e-6    | 0.08   | 0.0020    | 0.0003 → 0.00005 |
| V25     | 9e-7    | 0.10   | 0.0025    | 5e-5 → 1e-5      |

We wanted the critic to adapt to the current opponent pool while keeping actor updates more conservative. We therefore clipped actor and critic gradient norms separately, at 0.28 and 0.5, respectively.

V25 also used an L2 penalty with coefficient 0.025 between the actor’s parameters and a frozen reference checkpoint, intended to preserve previously measured general strength during fine-tuning. Backpropagation through time used `seq_len = n_steps = 128`, covering up to 128 turns within an episode, with recurrent state reset at episode boundaries.

Other settings included `vf_coef = 0.5`, disabled value-function clipping, fused Adam, float16 automatic mixed precision, the `channels_last` memory layout, and TF32.

## Inference

We deployed the actor through a Python inference script. Each API request is converted directly into the same nine-channel observation used during training, and PyTorch computes the action logits.

The agent then applies safety checks before selecting an action. A hard mask excludes moves that are known to cause immediate death (such as wall collisions). It also accounts for delayed growth due to collecting food to correctly determine the position of the tail in the next turn. A second tier of action masking avoids losing head-to-head collisions whenever safer alternatives exist.

The highest-scoring remaining action is then checked for forced self-traps using a bounded depth-first search, looking up to 18 turns ahead. Each candidate action has a budget of 12,000 search nodes. The search models our own body movement, health, food consumption, and delayed growth. It does not predict future opponent movements, so surviving the search horizon is not a guarantee of safety against other snakes.

If the preferred action has a continuation that reaches the search horizon, it is kept. Exhausting the node budget produces an **unknown** result, not proof of a trap, so the preferred action is also retained in that case. If the search proves that it leads to a self-trap, alternatives are examined and the highest-scoring remaining candidate is selected. If no action passes the action mask, the agent falls back to its highest-scoring action.

## Results

Der Snaketürke finished seventh out of 66 entries in the first round and seventh out of eight in the second round.

Our win rates against the other seven finalists followed their final ranking: we lost most often against the strongest opponent, next most often against the second-strongest, and so on. This pattern was consistent with our goal of developing broadly competitive behavior, but it does not by itself establish that the diverse opponent pool caused better generalization.

In many games, our snake initially avoided unnecessary food to remain short. Once only one opponent remained, it often circled within a particular area, occasionally leaving to collect nearby food before returning. In successful games, it maintained this pattern until the opponent ran into a dead end or a wall.

The main lessons from development were the importance of aligning rewards with tournament placement, training against varied strategies, and handling the game's collision and growth rules accurately during inference.

## Reflections on AI use in competitions

We relied heavily on AI agents during development, especially in the later stages, when parts of the project exceeded our own expertise. Their role extended beyond implementation: we also asked them to propose improvements and explore ideas with increasing autonomy. This is an important part of how the entry was developed, and the technical work described above should be read with that contribution in mind.

Our experience left us with questions about what competitions like Battlesnake measure as these tools become more capable. Performance may increasingly reflect a combination of a participant’s expertise, their ability to direct an agent, and the agent’s own capabilities. We suspect that knowing which problems to investigate and how to assess a proposed solution remains valuable, but our experience alone cannot establish how much human guidance improves results.

For us, the practical lesson is that delegating more work makes understanding and evaluation more important. Code that runs and a convincing explanation are not sufficient evidence that a change improves the agent. Retaining enough human understanding to question assumptions, interpret results, and redirect development remains an important goal, even as more of the implementation and experimentation becomes automated.
