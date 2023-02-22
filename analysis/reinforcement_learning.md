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

---

## The Bellmann Equation

```
Q(Sₜ,Aₜ) ← Q(Sₜ,Aₜ) + 𝛼 [ Rₜ₊₁ + 𝛾 maxQ(Sₜ₊₁,a) - Q(Sₜ,Aₜ) ]
```

- 𝜀-greedy action selection

## Implementation chapters

- Q network
  - state representation from last n state pictures feeding in a convolutional network
  - convolutional 
  - 
- Learning progress monitoring
  - log stats while learning
  - plot learning curve
