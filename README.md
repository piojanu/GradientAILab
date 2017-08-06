# AtariAI

In this project we want to explore reinforcement learning techniques in deep learning. Our baseline is experiment presented by Andrej Karpathy on his [blog](http://karpathy.github.io/2016/05/31/rl/). You can find his post in `<project root>/etc` directory. His work is about lightweight Policy Gradient agent.  
Starting from reproducing his work, we will further develop better agents for common reinforcement learning tasks and eventually start whole new projects based on gained experience.  
Along the road, article for Gradient research circle website should emerge.

# Repository organization

## Directory tree

```
.
├── doc (Article, presentations etc. related files.)
├── etc (Other files related to project e.g. papers.)
│   └── Deep Reinforcement Learning: Pong from Pixels.pdf
├── README.md (This file. Organization, targets, tasks, descritptions etc.)
└── src (All experiments live here.)
    ├── checkpoints (Saved models etc.)
    ├── codebase    (Classes, helpers, utils etc.)
    ├── logs        (All the logging related files.)
    ├── out         (All side products of scripts that don't fit anywhere else.)
    ├── third_party (As `codebase` but from third_party.)
    │   └── pg-pong.py
    └── script1.py  (All scripts performing experiments live in `src`.)
    
```

## [Mendeley](https://www.mendeley.com/)

There will be created group on Mendeley where collaborators will be able to share papers.

# Directions

After straightforward PG agent is done, we plan to explore those topics:

* [ ] Actor-critic agent:

    Reading:  
    * [Deterministic Policy Gradient Algorithms](http://proceedings.mlr.press/v32/silver14.pdf)
    * [Sample Efficient Actor-Critic with Experience Replay](https://arxiv.org/pdf/1611.01224.pdf)
      
* [ ] Imagination augmented agent:

    Reading:  
    * [Agents that imagine and plan](https://deepmind.com/blog/agents-imagine-and-plan/)
    * [Learning model-based planning from scratch](https://arxiv.org/pdf/1707.06170.pdf)
    * [Imagination-Augmented Agents for Deep Reinforcement Learning](https://arxiv.org/pdf/1707.06203.pdf)

---

_This is Gradient research circle project. Our website: http://gradient.eti.pg.gda.pl/_
