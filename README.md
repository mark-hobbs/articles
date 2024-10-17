# Design space exploration

A visual guide to design space exploration and optimisation

 <a href="https://colab.research.google.com/github/mark-hobbs/design-space-exploration/blob/main/design-space-exploration.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>

## Motivation

Efforts at the intersection of machine learning and simulation generally fall into two main categories: (1) accelerating simulations and (2) enhancing design space exploration and optimisation. These categories are interconnected, forming a feedback loop where advances in one area can significantly impact the other. Accelerating simulations can unlock *outer-loop* applications that are currently infeasible due to the high computational cost associated with numerous repeated simulations. Conversely, improving design space exploration can drastically reduce the number of simulations needed to identify optimal designs or strategically navigate the design space to minimise uncertainty.

## 1. Design space

Contour plot illustrating the relationship between two design parameters, $X_1$ and $X_2$, and the corresponding values of the objective function. In many engineering applications, evaluating this objective function can be extremely computationally expensive.

![](figures/design-space.png)

## 2. Grid search

Each point represents a single run of a computationally expensive simulation

![](figures/grid-search.png)

## 3. Monte Carlo (MC)

![](figures/monte-carlo.png)

## 4. Markov Chain Monte Carlo (MCMC)

MCMC samplers systematically sample the design space in a way that favours regions with higher probability or likelihood, effectively focusing the exploration on the most probable designs. This approach is particularly useful when the goal is to understand the distribution of optimal solutions or identify regions of the design space that meet specific criteria, rather than exhaustively searching all possibilities.

![](figures/mcmc.png)

## 5. Optimisation

The primary goal of optimisation is to find the best solution(s) according to specific objective criteria. The process inherently involves exploring the design space to identify regions that offer optimal performance. Techniques such as gradient-based optimisation, genetic algorithms, or Bayesian optimisation navigate the design space by sampling points, evaluating their performance, and iteratively refining the search. Thus, optimisation not only seeks the best designs but also contributes to understanding the structure and characteristics of the design space.

**Gradient-based optimisation**

![](figures/optimisation.png)

**Genetic algorithm**

![](figures/genetic-algorithm.gif)

## 6. Generative model

Generative models aim to infer the underlying distribution from which observed data is generated. In the engineering domain, models such as Generative Adversarial Networks (GANs) and Variational Autoencoders (VAEs) are employed to generate new design candidates by learning the distribution of existing high-performance designs. These models can create novel design configurations that retain desirable characteristics of known solutions.

![](figures/generative-model.png)