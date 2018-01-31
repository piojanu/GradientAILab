# Hello!

I'm really glad you're reading this, we value people interested in Deep Learning and Reinforcement Learning!

First of all, for now we collaborate mostly with Gradient student research circle members. If you are from Gdańsk, you can join us! Read more at http://gradient.eti.pg.gda.pl/o-nas/.  
**You should also familiraze with our values. Read CODE_OF_CONDUCT.md in root directory of this repo.**

# What contribution do we need?

From *reviewing projects and proposing some ideas* in issues to *coding and leading own projects*. Start small and then build a momentum. Also don't be intimidated, everybody was in your position at the beginning :) Learn with each commit/contribution!

# How to start?

Read README.md in root directory of this repo. It's best starting point.

# _(Not only)_ Coding standards

* **README**

    Each project has to have it's own README. Here you can find how to write one:
    * [Making READMEs readable](https://open-source-guide.18f.gov/making-readmes-readable/)
    * [README-Template.md by Billie Thompson](https://gist.github.com/PurpleBooth/109311bb0361f32d87a2)

* **Python**

    [PEP 8 -- Style Guide for Python Code](https://www.python.org/dev/peps/pep-0008/) is in operation.
    
    If you are emacs user, I recommend installing this package: py-autopep8. Configuration:  
    ```elisp
    ;; enable autopep8 formatting on save
    (require 'py-autopep8)
    (add-hook 'elpy-mode-hook 'py-autopep8-enable-on-save)
    ```  
    If you look for the best python/markdown/everything IDE and want to configure it easily, here is a guide for you: https://realpython.com/blog/python/emacs-the-best-python-editor/ and then http://jblevins.org/projects/markdown-mode/ .

* **Git commits**

    * [How to Write a Git Commit Message](https://chris.beams.io/posts/git-commit/) is in operation.

    * Commit name should start with capitalize project tag `[PROJECT]`. For now we have following tags:
        * [ORG] - "Organization" project e.g. change in this README.
        * [PG_PONG] - "Policy Gradient Pong" project based on Karpathy's blog post.
        * [Q_LEARN] - "Introduction to TD-Learning and Q-Learning algorithm" project.
        * [BIPEDAL] - Bipedal walker experiments project.  
		  
		You can install [this hook](https://gist.github.com/piojanu/4a68c70411f25f9bfcdee194d3dba374) to remind you about adding it.

    * If you work in this repo, remote branch names should follow those templates:

        * Dev branches: `dev/<project tag>/<user name>/<your branch name>`
        * Project branches: `proj/<project tag>/<branch name e.g. master>`

        Project branches will be merged to origin/master each milestone.
                
* **Pull requests**

    * If you want to commit to this repo: fork it, work on some project and then create a pull request.  
    **Pull request to master is mandatory even for collaborators!**
    
    * Also before creating pull request, squash your commits. It provides clarity in master branch history.
