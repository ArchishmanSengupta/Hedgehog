"""
Setup script for hedgehog package.
"""

from setuptools import setup, find_packages
import os

# Read version from __init__.py
def get_version():
    version_file = os.path.join(os.path.dirname(__file__), 'hedgehog', '__init__.py')
    with open(version_file, 'r') as f:
        for line in f:
            if line.startswith('__version__'):
                return line.split('=')[1].strip().strip('"').strip("'")
    return "0.1.0"

# Read requirements
def get_requirements():
    req_file = os.path.join(os.path.dirname(__file__), 'requirements.txt')
    if os.path.exists(req_file):
        with open(req_file, 'r') as f:
            return [line.strip() for line in f if line.strip() and not line.startswith('#')]
    return []

setup(
    name="hedgehog",
    version=get_version(),
    author="ArchishmanSengupta",
    author_email="senguptaarchie@gmail.com",
    description="A lightweight framework for training and fine-tuning Diffusion Language Models",
    long_description=open("README.md", "r", encoding="utf-8").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/hedgehog-dlm/hedgehog",
    packages=find_packages(exclude=["tests", "examples"]),
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: Apache Software License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Scientific/Engineering :: Machine Learning",
    ],
    python_requires=">=3.10",
    install_requires=get_requirements(),
    extras_require={
        "dev": [
            "pytest>=7.0",
            "black>=23.0",
            "ruff>=0.1.0",
            "mypy>=1.0",
        ],
        "train": [
            "accelerate>=0.20.0",
            "deepspeed>=0.10.0",
        ],
        "infer": [
            "vllm>=0.5.0",
            "sglang>=0.4.0",
            "lmdeploy>=0.5.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "hedgehog=hedgehog.cli:main",
        ],
    },
    include_package_data=True,
    package_data={
        "hedgehog": ["py.typed"],
    },
    zip_safe=False,
)
