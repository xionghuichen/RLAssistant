#!/usr/bin/env python
# coding=utf-8

from setuptools import setup, find_packages

setup(
        name='RLA',
        version="0.6.0",
        description=(
            'RL assistant'
        ),
        author='Xiong-Hui Chen',
        author_email='chenxh@lamda.nju.edu.cn',
        maintainer='Xiong-Hui Chen',
        packages=[package for package in find_packages()
                        if package.startswith("RLA")],
        platforms=["all"],
        install_requires=[
            "pyyaml",
            "argparse",
            "dill",
            "seaborn",
            "pathspec",
            'tensorboardX', 
            'pysftp',
            'typing'
        ]
    )
