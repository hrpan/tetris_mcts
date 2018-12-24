# Learning to play Tetris with Monte Carlo Tree Search and Temporal Difference Learning

My personal project for the love of Tetris.

(Warning: Codes are a hot mess riddled with inconsistent styles and unclear namingsm, read them at your own risk.)

## Introduction

This project started out as a practice to apply Deep Q-Learning to Tetris, one of my favourite puzzle games of all time. 
However, I soon realized that it was almost impossible to train an agent to perform anywhere near human level probably 
due to the sparsity and long-term dependency of the rewards in Tetris (imagine how many actions you need to perform to clear even one 
line!). It was also around that time AlphaGo beat Lee Sedol in a dominating fashion that reignited my hopes for a better agent.
And here it is, the MCTS-TD agent inspired by AlphaGo specializing in the game Tetris.


## Prerequisite

* torch==1.0.0 
* numpy==1.14.2
* numba==0.39.0
* tables==3.4.2
* matplotlib==2.1.2
* tensorflow==1.12.0 (not supported anymore, switch to PyTorch instead)

You'll also need the Tetris environment from here https://github.com/hrpan/pyTetris
and modify the `sys.path.append` in `play.py` to include the path of pyTetris.

## How to run it?

* `play.py` is a high level UI for self-play or manual play
* `train.py` is used for training the neural network

The default routine is written in `cycle.sh`, if you are unsure what to do simply use `./cycle.sh` and things should get going.

## Preliminary results
In the default procedure (`cycle.sh`), each iteration consists of 100 selfplay games with 300 MCTS simulations to generate the 
training data and 1 benchmark game with 1500 MCTS simulations to test the performance of the agent.

