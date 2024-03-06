#!/bin/bash

PYTHONPATH=$(pwd)/src pytest --nbmake --nbmake-timeout=12000 examples/*.ipynb papers/*.ipynb
