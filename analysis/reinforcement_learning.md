# Reinforcement learning (RL)

## Towards Q-learning | reinforcement learning concepts

### Stationary task | k-armed Bandit Problem 
_K choices with stationary reward distribution for each choice. Stationary means, the reward probabilities do not change over time._

__Incremental update of the Q value for a single step of one arm:__
```
             ₙ
Qₙ₊₁ = 1/n * ∑ Rᵢ 
             ⁱ⁺¹
Qₙ₊₁ = Qₙ + 1/n * [Rₙ - Qₙ]
```
---
__Generalized form:__
```
NewEstimate ← OldEstimate + StepSize [Target - OldEstimate]
```
`[Target - OldEstimate]`is an _error_ in the estimate

---

### Finite Markov Decision Processes (finite MDP)

The MDP framework is a considerable abstraction of the problem of goal-directed
learning from interaction.

Agent
: Learner and decision maker

Environment
: The thing the _agent_ interacts with, comprising everything outside the agent

- A - action (produced by agent)
- S - state (produced by environment)
- R - reward (produced by environment)
- t - time step
- trajectory - sequence: S₀, A₀, R₁, S₁, A₁, R₂, S₂, A₂, R₃

In a finite MDP, the sets of states, actions, and rewards (S, A, and R) all have a finite
number of elements.

Of course, the particular states and actions vary greatly from task to task, and how
they are represented can strongly affect performance. In reinforcement learning, as in
other kinds of learning, such representational choices are at present more art than science.

#### Reward signal

Represents what should be achieved (not how). It quantifies real results (positive and negative).

Discount rate 𝛾 (gamma)
: (0 <= 𝛾 <= 1) represents the value of future rewards. The bigger, the more farsighted the agent becomes

```
Gₜ = Rₜ₊₁ + 𝛾Rₜ₊₂ + 𝛾²Rₜ₊₃ ...
Gₜ = Rₜ₊₁ + 𝛾(Rₜ₊₂ + 𝛾Rₜ₊₃ + 𝛾²Rₜ₊₄)
Gₜ = Rₜ₊₁ + 𝛾Gₜ₊₁
```
---

### The Bellmann Equation

```
Q(Sₜ,Aₜ) ← Q(Sₜ,Aₜ) + 𝛼 [ Rₜ₊₁ + 𝛾 maxQ(Sₜ₊₁,a) - Q(Sₜ,Aₜ) ]
```


𝜀-greedy action selection
: (0 <= 𝜀 <= 1). The epsilon greedy parameter steers the probability of the action selection process at a certain step, 
to either use the current prediction of the model or to use a random value instead. 
It's important during the training phase to find a good balance between using what was learned, but also discover new options. 
The 𝜀-greedy parameter usually starts with 1.0 and is decreased during learning towards a defined minimum value.


## Implementation parts

- Q network
  - state representation from last n state pictures feeding in a convolutional network
  - convolutional 
  - 
- Learning progress monitoring
  - log stats while learning
  - plot learning curve
