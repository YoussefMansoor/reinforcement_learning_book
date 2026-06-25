# Chapter 4 – Monte Carlo Methods: The Empirical Learner

Author: [Youssef Mansour](www.linkedin.com/in/youssef-mansour-8b9609212)

---

> *"In theory, theory and practice are the same. In practice, they are not."*
> — Yogi Berra

So far in this book, we have been assuming something quite convenient: that we *know* the environment. In Dynamic Programming (Chapter 3), we needed the full transition model — every probability of every state transition, every reward — spelled out in advance. That is a strong assumption. A chess engine cannot enumerate every possible game position. A self-driving car cannot be handed a perfect map of every possible driver and pedestrian combination. Most interesting real-world problems refuse to hand over their rulebook.

This is where **Monte Carlo (MC) methods** come in. The key insight is almost philosophical in its simplicity:

> *If you want to know the expected outcome of something, just try it many times and average the results.*

No model needed. No transition matrix. Just experience — and a lot of it.

---

## What This Chapter Covers

This chapter builds up Monte Carlo methods from the ground up. Here is the road we will travel:

1. **The big picture (§4.1):** How learning from *experience* differs from planning with a *model*.
2. **Prediction (§4.2):** Estimating the value of a fixed policy by averaging observed returns — including the episode structure, the first-visit vs. every-visit distinction, and a full algorithm with code.
3. **A case study (§4.3):** The Racetrack task, to make the ideas concrete.
4. **From prediction to control (§4.4–4.5):** Why we estimate action-values $Q(s,a)$ instead of state-values, and how alternating policy evaluation and improvement drives us toward an optimal policy.
5. **The exploration problem (§4.6) and its three solutions (§4.7–4.10):** Exploring Starts, on-policy $\varepsilon$-greedy control, and off-policy control with importance sampling.
6. **Pitfalls and properties (§4.11–4.12):** The infinite-variance trap, and why Monte Carlo's lack of bootstrapping is a genuine advantage.
7. **Summary, exercises, and further reading (§4.13–4.15).**

By the end, you should be able to explain *why* Monte Carlo works, *when* to reach for it, and *how* to implement it.

---

## 4.1 The Map vs. The Journey

Before diving in, let's get clear on what makes Monte Carlo different from what we have seen before.

**Dynamic Programming** is like studying a road map before a trip. You have the complete layout — every intersection, every distance, every possible turn. With that map, you can compute the optimal route mathematically. It is powerful, but only if you actually have the map.

**Monte Carlo** is like taking the trip yourself, over and over, and building intuition through experience. You start somewhere, wander through the world according to some policy, collect rewards along the way, and reach a terminal state (the end of an *episode*). Then you look back at what happened and ask: *"Was that trip good or bad? What should I do differently next time?"*

This matters because:

- **No model required.** Monte Carlo works directly from samples of experience — real or simulated.
- **Only sample transitions are needed.** You don't need to know the probability of every possible next state; you just need to see *a* next state.
- **Episode-based.** Monte Carlo requires the task to be *episodic* — it must eventually end. (This is its main limitation: it cannot be applied to never-ending, continuing tasks without modification.)

![Side-by-side comparison: on the left, Dynamic Programming shown as a fully connected grid of states with transition-probability labels p(s'|s,a) on every edge; on the right, Monte Carlo shown as the same grid but faded, with a single red sampled trajectory running from a green start state S0 through intermediate states to a red terminal state, collecting rewards R1, R2, R3, R4 along the way.](figures/dp_vs_mc.png)
*Figure 4.1: The contrast at a glance. **Dynamic Programming (left)** needs the entire model — every transition probability $p(s'\mid s,a)$ and reward must be known up front, so the whole graph is "lit up." **Monte Carlo (right)** ignores the model entirely: it simply follows one episode at a time (the red path), observes the rewards that actually occur, and learns by averaging those observed returns. The faded edges underscore that MC never needs to know the transitions it didn't take.*

---

## 4.2 Monte Carlo Prediction: Learning Value Functions from Experience

### 4.2.1 The Core Idea

Recall from earlier chapters that the **value of a state** under a policy $\pi$ is the expected total return (sum of future rewards) starting from that state:

$$v_\pi(s) = \mathbb{E}_\pi\!\left[G_t \mid S_t = s\right]$$

where $G_t$ is the **return** from time $t$ onwards:

$$G_t = R_{t+1} + \gamma R_{t+2} + \gamma^2 R_{t+3} + \cdots$$

Monte Carlo prediction estimates this expectation by a beautifully simple mechanism: **run episodes, observe the returns $G_t$, and average them**. By the Law of Large Numbers, as we collect more and more sampled returns for a given state, their average converges to the true expected value $v_\pi(s)$.

### 4.2.2 The Episode and How We Compute Returns

An episode is a complete trajectory through the MDP, from start to termination:

$$S_0, A_0, R_1, S_1, A_1, R_2, S_2, \ldots, S_{T-1}, A_{T-1}, R_T, S_T$$

where $S_T$ is the **terminal state** and $T$ is the **terminal time step**.

The definition of $G_t$ above is a *forward* sum, but computing it that way for every $t$ would be wasteful — we'd redo overlapping work. Instead, notice that the return obeys a simple recursive relationship:

$$G_t = R_{t+1} + \gamma\, G_{t+1}.$$

In words: *the return at time $t$ is the immediate reward plus the discounted return from the next step.* This lets us compute every $G_t$ in a single **backward sweep**, starting from the end of the episode (where $G_T = 0$, because nothing follows the terminal state) and walking back to the start:

```text
G_T     = 0                      # nothing follows the terminal state
G_{T-1} = R_T   + γ · G_T        # = R_T
G_{T-2} = R_{T-1} + γ · G_{T-1}
   ⋮
G_t     = R_{t+1} + γ · G_{t+1}  # general recursive step
```

**A concrete example.** Suppose an episode lasts $T = 3$ steps with rewards $R_1 = 2$, $R_2 = 0$, $R_3 = 5$, and discount $\gamma = 0.9$. Sweeping backward:

$$
\begin{aligned}
G_3 &= 0 \\
G_2 &= R_3 + \gamma G_3 = 5 + 0.9(0)   = 5 \\
G_1 &= R_2 + \gamma G_2 = 0 + 0.9(5)   = 4.5 \\
G_0 &= R_1 + \gamma G_1 = 2 + 0.9(4.5) = 6.05
\end{aligned}
$$

So the observed return from the start state is $G_0 = 6.05$. This single backward pass gives us the return from *every* time step at once — which is exactly what the Monte Carlo algorithm needs.

### 4.2.3 First-Visit vs. Every-Visit MC

Here is a subtle question: what if the agent visits the same state $s$ **more than once** in a single episode? How many of those visits should contribute to our estimate of $v_\pi(s)$?

Two conventions exist:

| Method | Rule |
|---|---|
| **First-Visit MC** | For each episode, use only the return that follows the *first* time $s$ is visited. |
| **Every-Visit MC** | For each episode, use the return that follows *every* visit to $s$. |

**A worked example.** Take a single episode that visits state $s$ twice. Let the trajectory of states be $s \rightarrow x \rightarrow s \rightarrow (\text{terminal})$, with rewards $R_1 = 1$, $R_2 = 2$, $R_3 = 3$ and $\gamma = 1$ for simplicity. State $s$ appears at time steps $t = 0$ and $t = 2$.

First we compute the returns with the backward sweep:

$$G_3 = 0, \quad G_2 = 3, \quad G_1 = 2 + 3 = 5, \quad G_0 = 1 + 5 = 6.$$

Now look up the two visits to $s$:

- The **first visit** is at $t = 0$, with return $G_0 = 6$.
- The **second visit** is at $t = 2$, with return $G_2 = 3$.

The two methods treat these differently:

- **First-Visit MC** records only $G_0 = 6$ for this episode (it ignores the later visit).
- **Every-Visit MC** records *both* $G_0 = 6$ and $G_2 = 3$.

Across many episodes, each method simply averages all the returns it has recorded for $s$. For instance, if three separate episodes gave first-visit returns of $6$, $4$, and $8$, then

$$\hat{v}_\pi(s) = \frac{6 + 4 + 8}{3} = 6.$$

Both methods converge to the true value $v_\pi(s)$ as the number of recorded returns grows. **First-Visit MC is unbiased.** Every-Visit MC is biased for finite samples but also converges asymptotically — and it is often preferred in practice because it extends more naturally to function-approximation settings, which we meet later in the book.

### 4.2.4 The Algorithm

Here is the First-Visit MC prediction algorithm in pseudocode. The comments explain what each step is doing:

```text
Initialize:
    V(s)       ← arbitrary value, for all states s        # our estimate, to be refined
    Returns(s) ← empty list,      for all states s        # all returns seen for s so far

Loop forever (one pass per episode):
    Generate an episode following π:                      # roll out the policy until termination
        S₀, A₀, R₁, ..., S_{T-1}, A_{T-1}, R_T
    G ← 0                                                 # running return, starts at the end
    Loop for t = T-1, T-2, ..., 0:                        # sweep BACKWARD through the episode
        G ← γ·G + R_{t+1}                                 # recursive return: G_t = R_{t+1} + γ·G_{t+1}
        If S_t does NOT appear in {S₀, ..., S_{t-1}}:     # "first-visit" check
            Append G to Returns(S_t)                      # record this return for state S_t
            V(S_t) ← average(Returns(S_t))                # estimate = mean of all recorded returns
```

And here is a clean Python implementation. Every block is annotated so you can map it back to the pseudocode:

```python
import numpy as np
from collections import defaultdict

def first_visit_mc_prediction(env, policy, num_episodes, gamma=1.0):
    """
    Estimate the state-value function v_pi for a given policy
    using First-Visit Monte Carlo Prediction.

    Args:
        env:          a Gymnasium-style episodic environment
        policy:       a callable, policy(state) -> action
        num_episodes: number of episodes to sample
        gamma:        discount factor (gamma in [0, 1])

    Returns:
        V: dict mapping state -> estimated value
    """
    # Running sum and count of returns let us compute the mean incrementally.
    returns_sum = defaultdict(float)   # total return accumulated per state
    returns_count = defaultdict(int)   # how many returns we've recorded per state
    V = defaultdict(float)             # the value estimate we return

    for _ in range(num_episodes):
        # --- 1. Generate one full episode by following the policy --------
        episode = []                                  # list of (state, action, reward)
        state, _ = env.reset()
        done = False
        while not done:
            action = policy(state)                    # pick action from the fixed policy
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated            # episode ends on either flag
            episode.append((state, action, reward))
            state = next_state

        # --- 2. Sweep BACKWARD to compute returns (First-Visit MC) -------
        states_visited = set()                        # tracks first visits within THIS episode
        G = 0.0
        for t in reversed(range(len(episode))):       # t = T-1, ..., 0
            state_t, _, reward_t1 = episode[t]
            G = gamma * G + reward_t1                  # G_t = R_{t+1} + gamma * G_{t+1}

            if state_t not in states_visited:         # first-visit check
                states_visited.add(state_t)
                returns_sum[state_t] += G             # record this return
                returns_count[state_t] += 1
                V[state_t] = returns_sum[state_t] / returns_count[state_t]   # update mean

    return V
```

> **Practical note:** Libraries like [Gymnasium](https://gymnasium.farama.org/) (successor to OpenAI Gym) and [Stable-Baselines3](https://stable-baselines3.readthedocs.io/) provide environments well-suited for Monte Carlo experiments. For teaching purposes, `gym.make('Blackjack-v1')` is a classic testbed — it is episodic, has a manageable state space, and requires no model.

With prediction in hand, let's ground it in a problem you can picture.

---

## 4.3 Case Study: The Racetrack Task

Let's make this concrete with a case study from Sutton & Barto's textbook — the **Racetrack task**.

**Setup:** Imagine a discrete grid racetrack. A car starts on the starting line and must reach the finish line as quickly as possible without going out of bounds.

![A discrete grid racetrack drawn as a turning L-shaped corridor. The bottom row is the green START LINE, the right edge of the top section is the red FINISH LINE, and dark cells are out-of-bounds walls. A blue trajectory shows a car accelerating up the corridor, rounding the corner, and reaching the finish, with arrows indicating its changing velocity (vx, vy) at each step.](figures/racetrack.png)
*Figure 4.2: The Racetrack task. The car (green dot) begins on the start line with zero velocity and must reach the finish line (red) in as few steps as possible while staying on the track (light cells) and avoiding the walls (dark cells). Its **state** is its grid position together with its current velocity $(v_x, v_y)$; each **action** nudges that velocity. Monte Carlo learns a fast racing line purely by driving the car many times and averaging the returns — it never needs to know the car's physics.*

- **State:** Current grid position + current velocity (horizontal $v_x$, vertical $v_y$).
    - Velocity components: non-negative, each $< 5$, not both zero (except at the start).
- **Actions:** Increment or decrement each velocity component by $\pm 1$, or leave it unchanged → **9 possible actions**.
    - Complication: with probability $0.1$, the intended velocity change fails (environmental noise).
- **Rewards:** $-1$ per step (so minimizing steps maximizes return).
- **Termination:** Episode ends when the car crosses the finish line.
- **Boundary crash:** Car resets to a random start position with zero velocity; the episode *continues*.

**What does MC learning look like here?**

1. Generate many sample trajectories (drive the car from the start, following some policy).
2. At the end of each trajectory, compute the total return (the sum of $-1$ penalties, i.e. the negative episode length).
3. For each state-action pair $(s, a)$ encountered, record the return that followed it.
4. Average those returns to estimate $Q(s, a)$.
5. Improve the policy by choosing, in each state, the action with the highest estimated $Q$-value.

This is learning by doing, not by solving equations. And crucially — you don't need to know the transition probabilities or the physics of the car. You just simulate it.

Notice that step 4 above estimated $Q(s, a)$, not $V(s)$. That choice is not incidental, and the next section explains why it is forced upon us.

---

## 4.4 From Prediction to Control: Why We Need Q-Values

Prediction is great, but ultimately we want *control* — an optimal policy for acting. How does value estimation help us improve a policy?

### 4.4.1 The Problem with State Values Alone

If we only estimate the state values $V(s)$, we face a problem: to improve our policy, we need to choose actions. But without a model, we cannot look ahead to see which action leads to the best next state. The greedy improvement step would require

$$\pi'(s) = \arg\max_a \sum_{s', r} p(s', r \mid s, a)\left[r + \gamma V(s')\right],$$

which depends on $p(s', r \mid s, a)$ — the very transition probabilities we don't have.

### 4.4.2 Action-Value Functions to the Rescue

Instead, if we estimate $Q(s, a)$ — the action-value function — we can improve our policy *without a model*:

$$\pi'(s) = \arg\max_a Q(s, a).$$

We just pick the action with the best estimated $Q$-value. No model needed. This is why Monte Carlo control focuses on estimating $Q$, not $V$.

The MC algorithm for action values is identical in structure to the state-value version of §4.2.4 — just replace "first visit to state $s$" with "first visit to state-action pair $(s, a)$," and store returns in $Q(s,a)$ instead of $V(s)$.

---

## 4.5 Monte Carlo Control: Evaluating and Improving the Policy

Control works by interleaving two processes that push against each other:

- **Policy evaluation** — make the value estimate ($Q$ here) more accurate for the *current* policy.
- **Policy improvement** — make the policy greedy with respect to the *current* value estimate.

In Monte Carlo control, the evaluation step uses **sampled episodes** instead of model-based computation. We never run either process to completion before switching to the other; we simply alternate them, and as long as both keep improving, the policy and its value estimate converge together:

```text
          ┌─────────────────────────────────────────────┐
          │                                             │
   Generate episodes    ◄────────   Improve the policy   │
   by following π                   π ← greedy(Q)        │
          │                              ▲               │
          ▼                              │               │
   Evaluate: update Q(s,a)   ────────────┘               │
   from the observed returns                             │
          │                                             │
          └─────────────────────────────────────────────┘
```

After each episode:

1. **Policy evaluation:** Update the $Q$ estimates using the observed returns.
2. **Policy improvement:** Make the policy greedy with respect to the current $Q$.

This alternation — episode by episode — converges toward the optimal action-value function $Q^*$ and the optimal policy $\pi^*$.

But there is a hidden assumption lurking in the improvement step, and confronting it leads to the central tension of the rest of this chapter.

---

## 4.6 The Exploration-Exploitation Dilemma

Here is where things get interesting — and problematic.

If you always follow a *greedy* policy (always pick the action with the highest estimated $Q$-value), you will only ever explore actions you already think are good. You will never discover better alternatives. Your $Q$ estimates for the non-greedy actions will never improve, because you never take those actions.

This is the famous **exploration-exploitation tradeoff**:

> **Exploit:** Use what you know to get good rewards now.
> **Explore:** Try new things to learn whether something better exists.

If you only exploit, you risk converging to a *suboptimal* policy — forever stuck in a local optimum.

There are three main solutions to this problem in the MC setting. Let's walk through each in turn.

---

## 4.7 Solution A: Exploring Starts (MC-ES)

**The idea:** Guarantee that every state-action pair gets visited infinitely often by starting each episode from a *randomly chosen* pair $(s, a)$, with every pair having non-zero probability of being selected.

This is called the **Exploring Starts** assumption.

```text
MC with Exploring Starts:
Initialize:
    Q(s,a) ← arbitrary, for all s, a
    π(s)   ← greedy with respect to Q

Loop forever (one pass per episode):
    Choose S₀, A₀ at random  (every pair has probability > 0)   # the "exploring start"
    Generate an episode from (S₀, A₀) following π
    For each (s, a) that appears in the episode (first visit):
        G ← return following the first occurrence of (s, a)      # backward-sweep return
        Append G to Returns(s, a)
        Q(s, a) ← average(Returns(s, a))                        # policy evaluation
        π(s)    ← argmax_a Q(s, a)                              # policy improvement (greedy)
```

The policy improvement theorem guarantees that this converges to the optimal policy (informally — a full convergence proof for MC-ES remains an open problem in the field!).

**The catch:** Exploring Starts is often unrealistic. You cannot always start a self-driving car mid-crash just to ensure a specific starting state-action pair is covered. You cannot start a blackjack game mid-hand whenever you feel like it.

Put bluntly: *you cannot ask a self-driving car to start a learning episode "mid-crash."* Theoretically sound; practically impossible. So we need exploration that doesn't depend on controlling the start.

---

## 4.8 Solution B: On-Policy Control with ε-Greedy Policies

Instead of controlling the starting conditions, what if we build exploration *into the policy itself*?

**$\varepsilon$-greedy policy:** Most of the time (with probability $1 - \varepsilon$) take the greedy action, but with probability $\varepsilon$ take a *random* action. This ensures every action is tried occasionally. Formally:

$$
\pi(a \mid s) =
\begin{cases}
1 - \varepsilon + \dfrac{\varepsilon}{|\mathcal{A}(s)|} & \text{if } a = \arg\max_{a'} Q(s, a') \\[2mm]
\dfrac{\varepsilon}{|\mathcal{A}(s)|} & \text{otherwise}
\end{cases}
$$

This is an **on-policy** method: the same policy that generates the data is the one being improved.

```python
def epsilon_greedy_policy(Q, state, epsilon, n_actions):
    """Return an action drawn from an epsilon-greedy policy over Q."""
    if np.random.random() < epsilon:
        return np.random.randint(n_actions)   # explore: pick a uniformly random action
    else:
        return np.argmax(Q[state])            # exploit: pick the current best action
```

The full control loop uses this $\varepsilon$-greedy policy in both roles — generating the data and being improved:

```text
On-Policy First-Visit MC Control:
Initialize:
    Q(s,a) ← arbitrary
    π      ← an ε-soft policy (every action has probability ≥ ε/|A|)

Loop forever (one pass per episode):
    Generate an episode following π
    G ← 0
    Loop for t = T-1, ..., 0 (first-visit check on (S_t, A_t)):
        G ← γ·G + R_{t+1}                              # backward-sweep return
        Q(S_t, A_t) ← average of returns for (S_t, A_t)  # policy evaluation
        A* ← argmax_a Q(S_t, a)                         # the current greedy action
        For all actions a:                              # policy improvement: make π ε-greedy
            π(a | S_t) ← 1 - ε + ε/|A|   if a = A*
                          ε/|A|           otherwise
```

**The catch:** On-policy methods learn the best *$\varepsilon$-soft* policy — the best policy *that still explores*. They can never fully commit to a deterministic optimal policy, because they must keep exploring. Think of it as always hedging your bets: you learn the best strategy for someone who occasionally makes random decisions, not the absolute best strategy for someone who always decides optimally.

What if we could explore freely *and* still learn the fully greedy optimal policy? That is exactly what the third solution delivers.

---

## 4.9 Solution C: Off-Policy Control with Importance Sampling

Here is the most powerful and elegant solution. What if we could **separate the policy we learn about from the policy we use to gather data**?

- **Behavior policy** $b$: the policy that actually takes actions and generates episodes. It can be wild and exploratory — maybe even random.
- **Target policy** $\pi$: the policy we are trying to learn. It can be deterministic and greedy.

This is called **off-policy learning**, and it cleanly resolves the exploration-exploitation dilemma: the behavior policy explores; the target policy exploits and improves.

**Requirement (Coverage):** for off-policy learning to work, every action the target policy might take must also have a non-zero probability under the behavior policy:

$$\pi(a \mid s) > 0 \implies b(a \mid s) > 0, \quad \text{for all } (s, a).$$

Otherwise the target policy might "want" to take actions the behavior policy never explores, leaving us with zero data about them.

### 4.9.1 Importance Sampling: Correcting for the Wrong Distribution

The data we have comes from policy $b$, but we want to estimate values under policy $\pi$. The returns we observe therefore have the *wrong* expectation — they reflect $b$, not $\pi$.

**Importance Sampling** is a general statistical technique for estimating an expectation under one distribution ($\pi$) using samples drawn from another ($b$). The trick is to weight each return by how much more (or less) likely its trajectory was under $\pi$ compared to $b$.

The **importance-sampling ratio** for a trajectory running from time $t$ to $T$ is the product of the per-step policy ratios:

$$\rho_{t:T-1} = \prod_{k=t}^{T-1} \frac{\pi(A_k \mid S_k)}{b(A_k \mid S_k)}$$

Here is the beautiful cancellation: the full trajectory probabilities *also* depend on the environment's transition probabilities $p(s' \mid s, a)$ — but these appear identically in the numerator and denominator, so they cancel. The ratio depends only on the two policies, never on the unknown MDP dynamics.

**Intuition.** Reading the ratio $\rho$:

- $\rho > 1$ → trajectory **more** likely under $\pi$ than $b$ → weight it **up**.
- $\rho < 1$ → trajectory **less** likely under $\pi$ than $b$ → weight it **down**.
- $\rho = 0$ → trajectory could *never* happen under $\pi$ → **discard it**.

### 4.9.2 Ordinary vs. Weighted Importance Sampling

Two variants exist, with a classic bias-variance tradeoff. Let $\mathcal{T}(s)$ denote the set of time steps at which state $s$ is visited (across all episodes), and let $T(t)$ be the termination time of the episode containing step $t$.

**Ordinary Importance Sampling** divides the weighted returns by the *count* of visits:

$$V(s) = \frac{\displaystyle\sum_{t \in \mathcal{T}(s)} \rho_{t:T(t)-1}\, G_t}{\bigl|\mathcal{T}(s)\bigr|}$$

- Unbiased: $\mathbb{E}[V(s)] = v_\pi(s)$.
- High variance — can even be *infinite* variance, especially when trajectories involve loops.

**Weighted Importance Sampling** divides instead by the *sum of the weights*:

$$V(s) = \frac{\displaystyle\sum_{t \in \mathcal{T}(s)} \rho_{t:T(t)-1}\, G_t}{\displaystyle\sum_{t \in \mathcal{T}(s)} \rho_{t:T(t)-1}}$$

- Biased initially (but the bias $\to 0$ asymptotically).
- Much lower, *bounded* variance.
- **Strongly preferred in practice.**

![A line chart titled "The Risk: Infinite Variance." The green "Ordinary" importance-sampling estimate spikes wildly and erratically across episodes, while the red "Weighted" estimate stays low and flat. An inset shows the small looping Racetrack MDP that produces these looping trajectories.](figures/weighted_vs_ordinary.png)
*Figure 4.3: Why practitioners reach for weighted importance sampling. On a small looping MDP (inset), **ordinary IS (green)** produces violent, large-amplitude oscillations — its variance can be unbounded because a single rare high-$\rho$ trajectory can dominate the average. **Weighted IS (red)** stays smooth and stable, trading a small early bias for dramatically lower variance.*

Here is a concrete Python implementation of both variants, fully commented:

```python
def off_policy_mc_prediction(env, target_policy, behavior_policy,
                             num_episodes, gamma=1.0, weighted=True):
    """
    Off-policy MC prediction using Ordinary or Weighted Importance Sampling.

    Args:
        target_policy:   deterministic policy, target_policy(state) -> action
        behavior_policy: stochastic policy, behavior_policy(state) -> (action, prob)
        weighted:        if True use Weighted IS, otherwise Ordinary IS

    Returns:
        V: estimated value function under target_policy
    """
    numerator = defaultdict(float)     # sum of rho * G   (weighted numerator)
    denominator = defaultdict(float)   # sum of rho       (weighted denominator)
    count = defaultdict(int)           # visit count      (ordinary denominator)
    V = defaultdict(float)

    for _ in range(num_episodes):
        # --- 1. Generate an episode using the BEHAVIOR policy b ----------
        episode = []
        state, _ = env.reset()
        done = False
        while not done:
            action, action_prob = behavior_policy(state)   # b(a|s) and its probability
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            episode.append((state, action, reward, action_prob))
            state = next_state

        # --- 2. Backward sweep, accumulating return G and weight W -------
        G = 0.0
        W = 1.0                                            # importance weight rho, built up step by step
        for t in reversed(range(len(episode))):
            state_t, action_t, reward_t1, b_prob = episode[t]
            G = gamma * G + reward_t1                       # G_t = R_{t+1} + gamma * G_{t+1}

            # Importance ratio for this step: pi(a|s) / b(a|s).
            # The target policy is deterministic, so pi(a|s) is 1 or 0.
            pi_action = target_policy(state_t)
            pi_prob = 1.0 if pi_action == action_t else 0.0
            W = W * (pi_prob / b_prob)

            if W == 0:
                break   # target policy would never take this action -> earlier steps contribute nothing

            if weighted:
                numerator[state_t] += W * G                # accumulate weighted return
                denominator[state_t] += W                  # accumulate weight
                V[state_t] = numerator[state_t] / denominator[state_t]
            else:
                # Ordinary IS: average of the weighted returns (incremental mean).
                count[state_t] += 1
                V[state_t] += (W * G - V[state_t]) / count[state_t]

    return V
```

### 4.9.3 Incremental Updates: No Need to Store All Returns

One practical concern: storing every return and recomputing the average from scratch is memory-intensive. We can instead update the weighted average *incrementally* after each return, with no need to store the past.

Let $C_n$ be the cumulative sum of importance weights for a state after $n$ returns, and $V_n$ the estimate after $n$ returns. Then:

$$V_{n+1} = V_n + \frac{W_n}{C_n}\bigl[G_n - V_n\bigr]$$

$$C_{n+1} = C_n + W_{n+1}$$

This has the same form as the classic incremental-mean update, but with an adaptive learning rate $W/C$ in place of $1/n$:

```python
# Incremental Weighted IS update (inside the backward loop):
C[state][action] += W                                          # accumulate the weight
Q[state][action] += (W / C[state][action]) * (G - Q[state][action])
#                    └── learning rate ──┘   └─ target − estimate ─┘
```

Notice the structure: **learning rate × (target − estimate)**. This is the fundamental pattern of RL updates — you will see it again and again throughout the book.

---

## 4.10 Off-Policy MC Control: Putting It All Together

Here is the complete off-policy MC control algorithm. It estimates the optimal policy $\pi^*$ using a soft behavior policy $b$:

```text
Initialize, for all s, a:
    Q(s,a) ← arbitrary
    C(s,a) ← 0                          # cumulative importance weights
    π(s)   ← argmax_a Q(s, a)           # the (deterministic, greedy) target policy

Loop forever (one pass per episode):
    b ← any soft policy (e.g. ε-greedy w.r.t. Q)            # behavior policy that explores
    Generate an episode using b:
        S₀, A₀, R₁, ..., S_{T-1}, A_{T-1}, R_T
    G ← 0
    W ← 1                                                    # importance weight

    For t = T-1, T-2, ..., 0:
        G ← γ·G + R_{t+1}                                   # backward-sweep return
        C(S_t, A_t) ← C(S_t, A_t) + W                       # accumulate weight
        Q(S_t, A_t) ← Q(S_t, A_t) + (W / C(S_t, A_t)) · [G - Q(S_t, A_t)]   # incremental update
        π(S_t) ← argmax_a Q(S_t, a)                         # policy improvement

        If A_t ≠ π(S_t):
            break          # behavior diverged from the greedy target -> ρ = 0 from here back, stop
        W ← W · (1 / b(A_t | S_t))                          # update weight (π(A_t|S_t) = 1 for greedy target)
```

Note the `break` condition: we only keep updating backward while the actions taken match what the target policy would have chosen. The moment the behavior policy diverges from the greedy target policy, the importance weight for all earlier time steps becomes zero — so there is nothing more to learn from those earlier states in this episode.

---

## 4.11 The Risk: Infinite Variance in Ordinary IS

This section deserves its own spotlight, because it catches many practitioners off guard.

**Example (from Sutton & Barto).** Suppose there is one nonterminal state $s$ and two actions, `right` and `left`:

- `right` → deterministic termination, reward $0$.
- `left` → with probability $0.9$ loop back to $s$; with probability $0.1$ terminate with reward $+1$.

The target policy $\pi$ always chooses `left`, giving true value $v_\pi(s) = 1$. The behavior policy $b$ chooses `right` or `left` with equal probability ($0.5$ each).

When we use **ordinary importance sampling**, the variance of the estimates is *infinite* — even though the expected value is correct. In practice the estimates swing wildly and may never settle, even after millions of episodes (this is exactly the green curve in Figure 4.3).

**Weighted IS**, by contrast, gives an estimate of exactly $1$ after the first successful episode (one that ended via `left`); all other episodes contribute weight zero.

The lesson: ordinary IS is mathematically cleaner (unbiased) but practically dangerous. **Use weighted IS in practice.** As the racetrack MSE experiment in Figure 4.3 shows: ordinary IS may start lower but oscillates extremely, while weighted IS starts slightly higher yet decays smoothly and reliably toward zero.

---

## 4.12 Independence of Estimates: A Key Advantage

One structural property of Monte Carlo methods deserves explicit mention, because it is easy to overlook.

In Dynamic Programming (Chapter 3), the estimate for state $s$ depends on the estimates for its successor states $s'$. An error in one state propagates to its neighbors through the Bellman equation. This is called **bootstrapping**.

Monte Carlo does **not bootstrap**. Each return $G_t$ is computed entirely from real rewards observed in the episode. The estimate of $V(s)$ does not depend on the estimate of any other $V(s')$.

This has several consequences:

1. **No error propagation** across states — errors stay local.
2. **Parallel computation** is possible — you can update estimates for different states independently.
3. **Subset focus** — you can estimate the value of just one state by generating episodes starting from it, ignoring all others. This is impractical in DP.

---

## 4.13 Summary: The Trade-offs at a Glance

Let's zoom out and look at the full landscape of MC control methods:

| Approach | Exploration Method | Learns | Limitation |
|---|---|---|---|
| **MC-ES (Exploring Starts)** | Random episode starts | Optimal policy | Unrealistic in practice |
| **On-Policy ($\varepsilon$-greedy)** | Built into the policy | Best $\varepsilon$-soft policy | Cannot learn the fully greedy optimum |
| **Off-Policy (IS)** | Behavior policy explores | Optimal (target) policy | Higher variance, slower convergence |

And for importance sampling specifically:

| IS Variant | Bias | Variance | Practical Preference |
|---|---|---|---|
| **Ordinary** | None (unbiased) | Can be infinite | Not recommended |
| **Weighted** | Small ($\to 0$ asymptotically) | Always finite | **Preferred** |

The big-picture message: *Monte Carlo methods let us solve complex tasks without understanding the physics of the world.* We trade the planner's perfect map for the explorer's hard-won experience — and, episode by episode, that experience is enough.

A limitation lingers, though: Monte Carlo must wait until an episode *ends* before it can learn anything from it. The next chapter, on **Temporal-Difference learning**, removes that restriction — blending the sampling idea of Monte Carlo with the bootstrapping idea of Dynamic Programming so the agent can learn from every single step.

---

## 4.14 Exercises

Try these exercises to check your understanding:

1. **Conceptual.** Explain in your own words why state-value estimates $V(s)$ are not sufficient for policy improvement in the model-free setting. What do we need instead?

2. **Implementation.** Implement First-Visit MC prediction for the `Blackjack-v1` environment from Gymnasium. After 500,000 episodes, plot the value function for states with and without a usable ace. Compare your result to Figure 5.1 in Sutton & Barto (2020).

3. **Exploration.** In an $\varepsilon$-greedy policy with $\varepsilon = 0.1$ and $|\mathcal{A}| = 4$ actions, what is the probability of taking the greedy action? What is the probability of taking a specific non-greedy action?

4. **Importance Sampling.** For a trajectory of length 3 where
    - $\pi$ always takes action `left` (probability $1$),
    - $b$ takes `left` with probability $0.5$ and `right` with probability $0.5$,
    - all three actions taken were `left`,

    compute the importance-sampling ratio $\rho_{0:2}$.

5. **Coding Challenge.** Implement the off-policy MC control algorithm for the Blackjack environment. Use a random behavior policy and a greedy target policy. Plot the optimal policy found (similar to Figure 5.2 in Sutton & Barto).

6. **Reflection.** Why might off-policy methods converge more slowly than on-policy methods? In what application scenarios would you prefer off-policy despite the variance cost?

---

## 4.15 Further Reading and Resources

- **Sutton & Barto (2020), Chapter 5** — The authoritative treatment of Monte Carlo methods in RL. Free online at [incompleteideas.net](http://incompleteideas.net/book/the-book.html).
- **Gymnasium Blackjack environment** — A great testbed for MC methods: `pip install gymnasium`.
- **Spinning Up in Deep RL (OpenAI)** — An excellent practical resource for RL algorithms.
- **Kandemir (2021), SDU Lecture Notes** — Compact and mathematically precise slides on MC methods.

---

## References

Sutton, R. S., & Barto, A. G. (2020). *Reinforcement Learning: An Introduction* (2nd ed.). MIT Press. Chapter 5: Monte Carlo Methods, pp. 91–140.

Kandemir, M. (2021). *Monte Carlo Methods in RL* [Lecture slides]. University of Southern Denmark, Department of Mathematics and Computer Science (IMADA).

Singh, S. P., & Sutton, R. S. (1996). Reinforcement learning with replacing eligibility traces. *Machine Learning*, 22(1-3), 123–158.

Precup, D., Sutton, R. S., & Dasgupta, S. (2001). Off-policy temporal-difference learning with function approximation. In *Proceedings of the 18th International Conference on Machine Learning (ICML)*, 417–424.

---

To cite this chapter, please use the following BibTeX:

```bibtex
@misc{mansour_2026_ReinforcementLearning,
  author       = {Mansour, Youssef},
  title        = {Reinforcement Learning: A Gentle Introduction, Chapter 4},
  year         = {2026},
  publisher    = {GitHub},
  howpublished = {\url{https://github.com/amrmsab/reinforcement_learning_book}},
  note         = {Accessed: April 30, 2026}
}
```
