from setuptools import setup, find_packages

setup(
    name="levy_ou_pinn",
    version="1.0.0",
    description="Physics-Informed Neural Network for Lévy-driven Ornstein-Uhlenbeck Process",
    author="Generated from Notebook",
    packages=find_packages(),
    install_requires=[
        "torch>=1.9.0",
        "numpy>=1.21.0",
        "matplotlib>=3.4.0",
    ],
    python_requires=">=3.7",
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
    ],
) 