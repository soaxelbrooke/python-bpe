# coding=utf-8

from setuptools import setup

with open('requirements.txt') as infile:
    dependencies = [line.strip() for line in infile if len(line) > 0]

setup(name='bpe',
      version='0.1.0',
      description='Byte pair encoding for graceful handling of rare words in NLP',
      url='https://github.com/soaxelbrooke/bpe',
      author='Stuart Axelbrooke',
      autorh_email='stuart@axelbrooke.com',
      license='MIT',
      packages=['bpe'],
      install_requires=dependencies,
      zip_safe=False)
