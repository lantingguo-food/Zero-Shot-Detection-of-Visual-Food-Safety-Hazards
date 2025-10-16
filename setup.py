"""
Setup script for Zero-Shot Food Safety Detection
"""

from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

with open("requirements.txt", "r", encoding="utf-8") as fh:
    requirements = [line.strip() for line in fh if line.strip() and not line.startswith("#")]

setup(
    name="food-safety-zsd",
    version="1.0.0",
    author="Lanting Guo, Xiaoyu Hu, Wenhe Liu, Yang Liu",
    author_email="harryliu@ieee.org",
    description="Zero-Shot Detection of Visual Food Safety Hazards via Knowledge-Enhanced Feature Synthesis",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/food-safety-zsd",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
    ],
    python_requires=">=3.8",
    install_requires=requirements,
    extras_require={
        "dev": [
            "pytest>=6.0",
            "black>=21.0",
            "flake8>=3.9",
            "isort>=5.9",
        ],
    },
    entry_points={
        "console_scripts": [
            "zsfdet-train=train:main",
            "zsfdet-test=test:main",
            "zsfdet-inference=inference:main",
        ],
    },
)
