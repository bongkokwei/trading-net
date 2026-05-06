"""Setup configuration for trading-net package."""

from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="trading-net",
    version="0.1.0",
    author="Your Name",
    description="Time-series prediction for financial data using neural networks",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/bongkokwei/trading-net",
    packages=find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.6",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.6",
    install_requires=[
        "numpy>=1.19.0",
        "scikit-learn>=0.24.0",
        "matplotlib>=3.3.0",
        "requests>=2.25.0",
        "keras>=2.4.0",
        "tensorflow>=2.4.0",
    ],
)
