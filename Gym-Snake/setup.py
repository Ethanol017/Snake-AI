from setuptools import setup, find_packages

setup(name='gym_snake',
      version='0.0.1',
      author="Satchel Grant",
      install_requires=['numpy', 'matplotlib'],
      packages=find_packages(exclude=['logs']),
      python_requires='>=3',
      )
