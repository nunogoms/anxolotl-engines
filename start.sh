pipenv lock
pyenv init
eval "$(pyenv init -)"
pyenv exec python -m venv .venv
source .env/bin/activate
pipenv shell
pipenv sync