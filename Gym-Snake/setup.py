from setuptools import setup, find_packages

setup(
    name="gym_snake",
    version="0.0.1",
    author="Satchel Grant",
    install_requires=["numpy", "gymnasium>=0.29"],
    extras_require={
        "render": ["matplotlib>=3.7"],
    },
    packages=find_packages(exclude=["logs"]),
    python_requires=">=3",
)
