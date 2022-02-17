#!/usr/bin/env python
# coding=utf-8

from setuptools import setup, find_packages

setup(
        name='RLA',
        version=0.4,
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
            "pyyaml<=5.4.1",
            "argparse",
            "dill",
            "seaborn",
            "pathspec"
        ]
    )
