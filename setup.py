#!/usr/bin/env python

"""The setup script."""

from setuptools import setup, find_packages

with open('README.rst') as readme_file:
    readme = readme_file.read()

requirements = [
    'scipy']

setup_requirements = [
    'pytest-runner']

test_requirements = [
    'pytest>=3']

setup(
    author="Joeri Hermans",
    author_email='joeri@peinser.com',
    python_requires='>=3.5',
    classifiers=[
        'Development Status :: 2 - Pre-Alpha',
        'Intended Audience :: Developers',
        'License :: OSI Approved :: BSD License',
        'Natural Language :: English',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.5',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
    ],
    description="A Python toolkit for simulation-based inference and the mechanization of science.",
    entry_points={
        'console_scripts': [
            'hypothesis=hypothesis.cli:main',
        ],
    },
    install_requires=requirements,
    license="BSD license",
    long_description=readme,
    include_package_data=True,
    keywords='hypothesis',
    name='hypothesis',
    packages=find_packages(include=['hypothesis', 'hypothesis.*']),
    setup_requires=setup_requirements,
    test_suite='tests',
    tests_require=test_requirements,
    url='https://github.com/JoeriHermans/hypothesis',
    version='0.4.0',
    zip_safe=False,
)
