#!/bin/bash

# install dependencies
pip install pytest nbmake

# run tests
PYTHONPATH=$(pwd)/src pytest --nbmake --nbmake-timeout=12000 examples/*.ipynb papers/*.ipynb
