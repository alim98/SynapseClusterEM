from setuptools import setup, find_packages

setup(
    name="synapse_cluster_em",
    version="0.1",
    packages=find_packages(),
    install_requires=[
        "numpy>=1.17",
        "pandas>=1.2",
        "torch>=2.5.1",
        "torchvision>=0.20.1",
        "scikit-learn>=1.6.0",
        "matplotlib>=3.10.0",
        "seaborn>=0.13.2",
        "umap-learn>=0.5.7",
        "plotly>=5.3.1",
        "imageio>=2.31.0",
        "tqdm>=4.67.1",
        "kaleido>=0.2.1",
        "openpyxl>=3.1.5"
    ],
) 