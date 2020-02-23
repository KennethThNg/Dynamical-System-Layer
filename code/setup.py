from setuptools import setup, Extension, find_packages
from setuptools.command.build_ext import build_ext


with open('requirements.txt') as f:
    requirements = f.read().splitlines()

NAME = 'DynamcalSystem'
VERSION = '1.0'
DESCRIPTION = 'Inner layer based on dynamical system for deep neural network architecure.'
KEYWORD = 'Dynamical system, machine learning, deep learning, data science'
MAINTAINER = 'Kenneth Nguyen'
MAINTAINER_EMAIL = 'kenneth.thang.nguyen@gmail.com'
CLASSIFIER = ['Programming Language :: Python',
              'Intended Audience :: Science/Research',
              'Intended Audience :: Developers',
              'Topic :: Software Development',
              'Topic :: Scientific/Engineering']
INSTALL_REQUIRE = requirements

setup(name=NAME,
      version=VERSION,
      description=DESCRIPTION,
      keywords=KEYWORD,
      maintainer=MAINTAINER,
      maintainer_email=MAINTAINER_EMAIL,
      classifiers=CLASSIFIER,
      install_requires = INSTALL_REQUIRE)

