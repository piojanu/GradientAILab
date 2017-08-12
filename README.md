# AtariAI

In this project we want to explore reinforcement learning techniques in deep learning. Our baseline is experiment presented by Andrej Karpathy on his [blog](http://karpathy.github.io/2016/05/31/rl/). You can find his post in `<project root>/etc` directory. His work is about lightweight Policy Gradient agent.  
Starting from reproducing his work, we will further develop better agents for common reinforcement learning tasks and eventually start whole new projects based on gained experience.  
Along the road, articles/presentations for Gradient research circle should emerge.

# Repository organization

## Directory tree

```
.
├── doc (Articles/presentations related files, experiments descriptions/results etc.)
│   └── project_dir (As `doc` but specific to some project.)
├── etc (Other files related to project e.g. papers.)
├── README.md (This file. Organization, targets, tasks, descritptions etc.)
└── src (All experiments live here.)
    ├── checkpoints (Saved models etc.)
    ├── codebase    (Classes, helpers, utils etc.)
    ├── logs        (All the logging related files.)
    ├── out         (All side products of scripts that don't fit anywhere else.)
    ├── third_party (As `codebase` + scripts but from third_party.)
    ├── script1.py  (All scripts performing experiments live in `src`.)
    └── project_dir (As `src` but specific to some project.)

```

## Coding standards

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

    * **Everyone have to create pull request (not push to master directly) so that anyone could do code review!**

    * Commit name should start with capitalize project tag `[PROJECT]`. For now we have following tags:
        * [ORG] - "Organization" project e.g. change in this README.
        * [PG_PONG] - "Policy Gradient Pong" project based on Karpathy's blog post.

    * Remote branch names should follow those templates:

        * Dev branches: `dev/<project tag>/<user name>/<your branch name>`
        * Project branches: `proj/<project tag>/<branch name e.g. master>`

        Project branches will be merged to origin/master each milestone.

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

* [ ] Run and leap - Terrain-Adaptive Locomotion Skills:

    Reading:
    * [Terrain-Adaptive Locomotion Skills Using Deep Reinforcement Learning](https://www.cs.ubc.ca/~van/papers/2016-TOG-deepRL/index.html)

---

_This is Gradient research circle project. Our website: http://gradient.eti.pg.gda.pl/_  
_The truth is, we will send fully automated "Prometeusz" space cruise to the Solaris in colonization mission! Maybe.._
