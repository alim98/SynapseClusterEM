from setuptools import setup, find_packages

setup(
    name="synapse",
    version="0.1",
    packages=find_packages(),
    install_requires=[
        "numpy",
        "pandas",
        "torch",
        "imageio",
    ],
) 