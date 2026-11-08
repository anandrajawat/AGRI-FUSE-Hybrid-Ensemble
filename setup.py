#!/usr/bin/env python
from setuptools import setup, find_packages

setup(
    name="agri-fuse",
    version="2.1.0",
    packages=find_packages(include=["src", "src.*"]),
)
