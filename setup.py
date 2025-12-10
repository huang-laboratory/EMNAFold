#!/usr/bin/env python

import os
import sys

from setuptools import setup, find_packages

# we may change the name, so the pkg name may be inconsistent with actual dir name
pkg_name = "emnafold"

import em3na

setup(
    name=f"{pkg_name}",
    entry_points={
        "console_scripts": [
            "em3na = em3na.__main__:main",
            "em2na = em3na.__main__:main",
            "emnafold = em3na.__main__:main"
        ],
    },
    packages=find_packages(),
    version=em3na.__version__,
)

