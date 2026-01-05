from setuptools import setup, find_packages

setup(
    name='GTra',
    version='1.0.0',
    description='Gene expression pattern-based trajectory inference',
    author='Jaeyeon Jang',
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    install_requires=[
        "numpy", "pandas", "scipy", "matplotlib", "seaborn", "scanpy",
        "scikit-learn", "tqdm"
    ],
)
