from setuptools import setup, find_packages
from codecs import open
from os import path

here = path.abspath(path.dirname(__file__))

# Get the long description from the README file
with open(path.join(here, 'README.md'), encoding='utf-8') as f:
    long_description = f.read()

setup(
    name='fairml-farm',  # Required
    version='0.1.0',  # Required

    description='Implementations and tools for simple fairness algorithms',  # Required
    long_description=long_description,  # Optional
    url='https://github.com/yo-shavit/fairml-farm',  # Optional
    author='Yonadav Shavit',  # Optional
    author_email='yonadav.shavit@gmail.com',  # Optional
    packages=find_packages(),  # Required
    install_requires=['numpy', 'tensorflow>=1.4', 'matplotlib>=2.1.1', 'sklearn',
                      'pandas>=0.21.0'],  # Optional
)
