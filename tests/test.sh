#!/bin/bash

# install dependencies
pip install pytest nbmake

# run tests
PYTHONPATH=$(pwd) pytest --nbmake --nbmake-timeout=12000 examples/*.ipynb papers/*.ipynb
