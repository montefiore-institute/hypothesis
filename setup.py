# ```cag``` is free software; you can redistribute it and\or modify it
# under the terms of the Revised BSD License; see LICENSE file for more details.

"""```cag``` setup file."""

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
    exclusions.append("cag/extension")
if not include_benchmarks:
    exclusions.append("cag/benchmark")

packages = find_packages(exclude=exclusions)

# Get the version string of cag.
with open(os.path.join("cag", "__init__.py"), "rt") as fh:
    _version = re.search(
        '__version__\s*=\s*"(?P<version>.*)"\n',
        fh.read()
    ).group("version")

# Module requirements.
_install_requires = [
    "numpy",
    "torch"
]

_parameters = {
    "install_requires": _install_requires,
    "license": "BSD",
    "name": "cag",
    "packages": packages,
    "platform": "any",
    "url": "https://github.com/montefiore-ai/cag/",
    "version": _version
}

setup(**_parameters)
