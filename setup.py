#! /usr/bin/env python
# -*- coding: utf-8 -*-

from setuptools import setup, find_packages

version_git = '0.1'

with open('HISTORY.rst') as history_file:
    history = history_file.read()

requirements = [
    'numpy',
    'protobuf >= 3.8.0',
    'six',
]

test_requirements = [
    'pytest',
    'matplotlib',
    'crc32c',
]

setup(
    name='nnabla_tensorboard',
    version=version_git,
    description='nnabla_tensorboard lets you watch Tensors Flow with NNabla',
    long_description=history,
    author='naibo-code',
    author_email='naibo.ha@gmail.com',
    url='https://github.com/naibo-code/nnabla_tensorboard',
    packages=find_packages(exclude=['tests']),
    include_package_data=True,
    install_requires=requirements,
    license='MIT license',
    zip_safe=False,
    classifiers=[
        'Development Status :: 2 - Pre-Alpha',
        'Intended Audience :: Developers',
        'License :: OSI Approved :: MIT License',
        'Natural Language :: English',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
    ],
    test_suite='tests',
    tests_require=test_requirements
)
