# Learning to play Tetris with Monte Carlo Tree Search and Temporal Difference Learning

My personal project for the love of Tetris.

See the agent in action [here](https://www.youtube.com/watch?v=EALo2GfZuYU)!

(Warning: Codes are a hot mess riddled with inconsistent styles and unclear namingsm, read them at your own risk.)

## Introduction

This project started out as a practice to apply Deep Q-Learning to Tetris, one of my favourite puzzle games of all time. 
However, I soon realized that it was almost impossible to train an agent to perform anywhere near human level probably 
due to the sparsity and long-term dependency of the rewards in Tetris (imagine how many actions you need to perform to clear even one 
line!). It was also around that time AlphaGo beat Lee Sedol in a dominating fashion that reignited my hopes for a better agent. Also,
I believed that a model based approach should improve significantly compared to model free approaches (Q learning, policy gradients etc.). So here it is, the MCTS-TD agent inspired by AlphaGo specializing in the game Tetris.


## Prerequisite

* torch==1.0.0 
* numpy==1.14.2
* numba==0.39.0
* tables==3.4.2
* matplotlib==2.1.2
* tensorflow==1.12.0 (not supported anymore, switch to PyTorch instead)

You'll also need the Tetris environment from [here](https://github.com/hrpan/pyTetris)
and modify the `sys.path.append` in `play.py` to include the path of pyTetris.

## How to run it?

* `play.py` is a high level UI for self-play or manual play
* `train.py` is used for training the neural network

The default routine is written in `cycle.sh`, if you are unsure what to do simply use `./cycle.sh` and things should get going.



## Results
In the default routine (`cycle.sh`), each iteration consists of 100 selfplay games with 300 MCTS simulations to generate the 
training data and 1 benchmark game with 1500 MCTS simulations to test the performance of the agent.

<img src="https://ucdcc7f4a7f73e5822a1f83f242d.dl.dropboxusercontent.com/cd/0/inline/AYHYe1W2D1LOPooO48szN5ij1D9C7PiwbgnvopzAVrLGp0NuUXZa3lFMbTseIlu-PEuxGIqXeLWzik3iaQ4bHgAIhH9SqjKk-L19OPalAQr876rLwGAFpdcFBKsqAUP0jEdumpDqmLbUqdGE--MWVEC9vEcangSnebMUf0rhmJwiJYKWwgeJv3xyXT9AJyNt9nE/file" width="400"/> <img src="https://ucc3d299ec4c1971c52783ec5207.dl.dropboxusercontent.com/cd/0/inline/AYEUvzSUDjucHSWBkSzOWEzbDxMUtuPYQi23jdeBxIp92Y-eEkF_woq7LP2l8Z-SW2nSzTDNygETCb-Fp3d6k54TuR9uObOfov609bLHMRq7in3Kk7tEW22Gs3qTXmlbi9XfxCRvC_JJo8vDzmfovlt0ihyIUfCvLrrDsDB3gSOkvFlhfzTG-Bs2Y_kHDvksNuY/file" width="400"/> 

Left one is the normal (300 simulations) selfplay, right one is the benchmark (1500 simulations) selfplay.

As can be seen in the graphs, the agent is still improving even after 13 iterations (1300 games), however, it takes more than 10 hours to finish one iteration on my potato so I had to terminate it early.
