#!/usr/bin/env python

"""The setup script."""

import glob
import matplotlib as mpl
import os
import shutil

from setuptools import setup, find_packages



with open('README.md') as readme_file:
    readme = readme_file.read()

def _load_requirements(file_name="requirements.txt", comment_char='#'):
    with open(file_name, 'r') as file:
        lines = [ln.strip() for ln in file.readlines()]
    reqs = []
    for ln in lines:
        # filer all comments
        if comment_char in ln:
            ln = ln[:ln.index(comment_char)].strip()
        # skip directly installed dependencies
        if ln.startswith('http'):
            continue
        if ln:  # if requirement is not empty
            reqs.append(ln)

    return reqs

setup_requirements = [
    'pytest-runner']

test_requirements = [
    'pytest>=3']

# Install the Matplotlib style files
style_dir = os.path.join(mpl.get_data_path(), 'stylelib')
style_files = glob.glob(os.getcwd() + "/mpl/*.mplstyle")
for f in style_files:
    filename = os.path.basename(f)
    destination = style_dir + '/' + filename
    shutil.copy(f, destination)

# Install Hypothesis
setup(
    author="Joeri Hermans",
    author_email='joeri@peinser.com',
    python_requires='>=3.8',
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
    install_requires=_load_requirements(),
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
