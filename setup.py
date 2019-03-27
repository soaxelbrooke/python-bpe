# coding=utf-8

from setuptools import setup

with open("README.md", "r") as infile:
    long_description = infile.read()

with open("requirements.txt") as infile:
    dependencies = [line.strip() for line in infile if len(line) > 0]

setup(
    name="bpe",
    version="1.0",
    description="Byte pair encoding for graceful handling of rare words in NLP",
    url="https://github.com/soaxelbrooke/python-bpe",
    author="Stuart Axelbrooke",
    author_email="stuart@axelbrooke.com",
    license="MIT",
    long_description=long_description,
    long_description_content_type="text/markdown",
    packages=["bpe"],
    setup_requires=["pytest-runner"],
    tests_require=["pytest", "hypothesis"],
    install_requires=dependencies,
    zip_safe=False,
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
)
