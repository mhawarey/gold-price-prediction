# setup.py
import setuptools
from setuptools import find_packages, setup
import os

# Read requirements from file
with open('requirements.txt') as f:
    requirements = f.read().splitlines()

# Package metadata
setup(
    name="gold_price_predictor",
    version="1.0.0",
    author="Gold Price Predictor Team",
    description="Advanced gold price prediction system",
    packages=find_packages(),
    python_requires=">=3.8",
    install_requires=requirements,
    entry_points={
        'console_scripts': [
            'gold-predictor=gold_price_predictor.main:main',
        ],
    },
)