# PG Pong

It's PyTorch implementation of [this](http://karpathy.github.io/2016/05/31/rl/) tutorial's example pg agent for pong game.

## Dependencies *(and version I used)*

* [Jupyter Notebook 5.0.0](http://jupyter.readthedocs.io/en/latest/install.html)
* [PyTorch 0.2.0](http://pytorch.org/)
* [OpenAI Gym 0.9.2](https://gym.openai.com/docs/)

## Usage

See `python pong_pg_play.py --help` and `python pong_pg_pytorch.py --help` for more information.

## Training

I've learned it for two days on Titan X Maxwell.
Learning it with lr 1e-3 for one day and then changing lr to 85e-5 for next day yielded the best result. It's average score is almost 8 (it wins most of the time!).

## Experiments
### Movement penalty

I tried to make it more calm (see how it plays to see what I mean...) by giving small negative reward for action UP or DOWN. Punishing for movement didn't help. It resulted in agent not moving at all.
Maybe penalty was to high (I used range from 0.001 to 0.00001).

## Future work

* Hiperparameters tuning
* ConvNets
* Actor-Critic method

---
_For information on Gradient AI Lab (group under which this project is developed) see main README.md in root of this repo._
