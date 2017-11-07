# AtariAI

In this project we want to explore reinforcement learning techniques in deep learning. Our baseline is experiment presented by Andrej Karpathy on his [blog](http://karpathy.github.io/2016/05/31/rl/). You can find his post in `<project root>/etc` directory. His work is about lightweight Policy Gradient agent.  
Starting from reproducing his work, we will further develop better agents for common reinforcement learning tasks and eventually start whole new projects based on gained experience.  
Along the road, articles/presentations for Gradient research circle should emerge.

## How to start AI&RL journey?

0. Of course some background in artificial neural networks to start with Deep Reinforcement Learning.
1. Artificial Intelligence grounding:
    * Berkeley cs188 course ([on edXedge](https://edge.edx.org/courses/course-v1%3ABerkeley%2BCS188%2BSP17/)):
        * [Uncertainty and Utilities](https://www.youtube.com/watch?time_continue=15&v=GevK0-9n24g)
        * [MDP part I](https://www.youtube.com/watch?v=Oxqwwnm_x0s&t=4034s) and [MDP part II](https://www.youtube.com/watch?v=6pBvbLyn6fE&t=847s)
        * [Reinforcement Learning part I](https://www.youtube.com/watch?v=IXuHxkpO5E8) and [Reinforcement Learning part II](https://www.youtube.com/watch?v=yNeSFbE1jdY)
    * Book [Artificial Intelligence: A Modern Approach (3rd Edition)](https://dcs.abu.edu.ng/staff/abdulrahim-abdulrazaq/courses/cosc208/Artificial%20Intelligence%20A%20Modern%20Approach%20(3rd%20Edition).pdf):
        * Ch. 16.1 - 16.3
        * Ch. 17.1 - 17.3
        * Ch. 21
2. Deep Reinforcement Learning:
    * Book [Reinforcement Learning: An Introduction (2nd Edition Draft)](http://ufal.mff.cuni.cz/~straka/courses/npfl114/2016/sutton-bookdraft2016sep.pdf)
    * Courses (personally I watch (and do assignments from) Deep RL Bootcamp and watch some talks from RL Summer School):
        * [Deep RL Bootcamp](https://sites.google.com/view/deep-rl-bootcamp/home)
        * [Reinforcement Learning Summer School](http://videolectures.net/deeplearning2017_montreal/)
        * [CS 294: Deep Reinforcement Learning](http://rll.berkeley.edu/deeprlcourse/)

## Repository organization

### _(Preferred)_ Directory tree

```
.
├── README.md (This file. Organization, targets, tasks, descritptions etc.)
├── etc (Other resources related to reinforcement learning in general e.g. papers)
└── <project name> 
    ├── README.md (Project description, organization, milestones etc.)
    ├── doc (Articles, presentations, experiments descriptions and results etc.)
    ├── etc (Other resources related to project e.g. papers, diagrams etc.)
    └── src (All experiments live here.)
        ├── checkpoints (Saved models etc.)
        ├── codebase    (Classes, helpers, utils etc.)
        ├── logs        (All the logging related files.)
        ├── out         (All side products of scripts that don't fit anywhere else.)
        ├── third_party (As `codebase` + scripts but from third party.)
        └── script1.py  (All scripts performing experiments live in `src`.)

```

### Coding standards

* **Python**

    [PEP 8 -- Style Guide for Python Code](https://www.python.org/dev/peps/pep-0008/) is in operation.
    
    If you are emacs user I recommend installing this package: py-autopep8. Configuration:  
    ```elisp
    ;; enable autopep8 formatting on save
    (require 'py-autopep8)
    (add-hook 'elpy-mode-hook 'py-autopep8-enable-on-save)
    ```  
    If you look for the best python/markdown/everything IDE and want to configure it easily, here is a guide for you: https://realpython.com/blog/python/emacs-the-best-python-editor/ and then http://jblevins.org/projects/markdown-mode/ .

* **Git commits**

    * [How to Write a Git Commit Message](https://chris.beams.io/posts/git-commit/) is in operation.

    * If you want to contribute to this repo: fork it, work on your project and then create a pull request. **Pull request is mandatory even for collaborators!**

    * Commit name should start with capitalize project tag `[PROJECT]`. For now we have following tags:
        * [ORG] - "Organization" project e.g. change in this README.
        * [PG_PONG] - "Policy Gradient Pong" project based on Karpathy's blog post.
        * [Q_LEARN] - "Introduction to TD-Learning and Q-Learning algorithm" project.

    * If you work in this repo, remote branch names should follow those templates:

        * Dev branches: `dev/<project tag>/<user name>/<your branch name>`
        * Project branches: `proj/<project tag>/<branch name e.g. master>`

        Project branches will be merged to origin/master each milestone.

## Directions

* [X] Stochastic Policy Gradients:

    Reading:
    * [Deep Reinforcement Learning: Pong from Pixels](http://karpathy.github.io/2016/05/31/rl/)

* [ ] Learning from Human Preferences:  
    
    Reading:
    * [OpenAI Blog: Learning from Human Preferences](https://blog.openai.com/deep-reinforcement-learning-from-human-preferences/)
    * [Learning from Human Preferences](https://arxiv.org/abs/1706.03741v3)

* [ ] Actor-critic agent:

    Reading:  
    * [Deterministic Policy Gradient Algorithms](http://proceedings.mlr.press/v32/silver14.pdf)
    * [Sample Efficient Actor-Critic with Experience Replay](https://arxiv.org/pdf/1611.01224.pdf)
    
    Extra for asynchronous actor-critic (A3C):
    * [Asynchronous Methods for Deep Reinforcement Learning](https://arxiv.org/abs/1602.01783v2)
      
* [ ] Imagination augmented agents and Schema Networks:

    Reading about "Imagination":  
    * [Agents that imagine and plan](https://deepmind.com/blog/agents-imagine-and-plan/)
    * [Learning model-based planning from scratch](https://arxiv.org/pdf/1707.06170.pdf)
    * [Imagination-Augmented Agents for Deep Reinforcement Learning](https://arxiv.org/pdf/1707.06203.pdf)

    Reading about "Schema Networks":
    * [General Game Playing with Schema Networks](https://www.vicarious.com/general-game-playing-with-schema-networks.html)
    * [Schema Networks: Zero-shot Transfer with a Generative Causal Model of Intuitive Physics](https://arxiv.org/abs/1706.04317)

* [ ] Run and leap - Terrain-Adaptive Locomotion Skills:

    Reading:
    * [Terrain-Adaptive Locomotion Skills Using Deep Reinforcement Learning](https://www.cs.ubc.ca/~van/papers/2016-TOG-deepRL/index.html)

---

_This is Gradient research circle project. Our website: http://gradient.eti.pg.gda.pl/_  
_The truth is, we will send fully automated "Prometeusz" space cruise to the Solaris in colonization mission! Maybe.._
