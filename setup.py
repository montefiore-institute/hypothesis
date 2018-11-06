# ```hypothesis``` is free software; you can redistribute it and\or modify it
# under the terms of the Revised BSD License; see LICENSE file for more details.

"""```hypothesis``` setup file."""

import os
import re
import sys

from setuptools import find_packages
from setuptools import setup

"""Configuration"""
include_extensions=True
include_benchmarks=True


exclusions=["doc", "examples"]
if not include_extensions:
    exclusions.append("hypothesis/extension")
if not include_benchmarks:
    exclusions.append("hypothesis/benchmark")

packages = find_packages(exclude=exclusions)

# Get the version string of hypothesis.
with open(os.path.join("hypothesis", "__init__.py"), "rt") as fh:
    _version = re.search(
        '__version__\s*=\s*"(?P<version>.*)"\n',
        fh.read()
    ).group("version")

# Module requirements.
_install_requires = [
    "numpy",
    "torch",
    "pkgutil"
]

_parameters = {
    "install_requires": _install_requires,
    "license": "BSD",
    "name": "hypothesis",
    "packages": packages,
    "platform": "any",
    "url": "https://github.com/montefiore-ai/hypothesis/",
    "version": _version
}

setup(**_parameters)
