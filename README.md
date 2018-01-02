# fairml-farm
This repository contains a collection of implementations of fair ML algorithms.
The goal is to have all of these approaches concentrated in the same place and directly comparable
to each other, so that we can more quickly compare new approaches against previous ones,
and compare and contrast the different algorithms. The ultimate goal is to provide a playground to help
build intuition about different definitions and approaches to fairness.

## Overview
The main components of this project are:

* *Implementations:* a set of implementations of popular fairness approaches, which we'll update periodically
* *Datasets:* functions to easily load popular fairness datasets
** *Synthetic Datasets* an interface for creating synthetic fairness datasets (w/ built-in visualizations)
* *Evaluation:* Side-by-side comparisons of different algorithms' behaviors across a slew of fairness metrics

For now, we'll be dealing exclusively with the classical fairness-in-classification problem setup:

Given data 'X', labels 'Yϵ{0, 1}', and protected attribute 'Aϵ{0,1}', we want to construct some classifier 'c(X)'
 which is both predictive of 'Y', and "fair" with respect to the two groups 'A=0' and 'A=1'.
 There are many definitions for such fairness, and each algorithm may be tuned to a different definition.

## Getting started

This project runs in Python 3 (developed in 3.5.2).
To get started, navigate to the project's root directory, and call:

'pip install -e .'

to install the various dependencies. Running all the project's code requires
 'numpy', 'tensorflow' (1.4.0 or later), 'matplotlib', 'pandas', and 'sklearn'.
 (The previous command will install these if you don't have them yet).

TODO: add a setup guide

### Installation
### Running the toy dataset example
## Adding a new implementation

## Table of Contents

## Future work
* Add module on fair representation learning algorithms
