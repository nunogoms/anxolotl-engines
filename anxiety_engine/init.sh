#! /bin/sh

pyenv init
eval "$(pyenv init -)"
pyenv exec python -m venv .venv
source .venv/bin/activate
pipenv shell
#Write 'exit' on pipenv shell
pipenv lock
pipenv sync