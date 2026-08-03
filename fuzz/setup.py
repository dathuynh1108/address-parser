from __future__ import annotations

import os

from Cython.Build import cythonize
from setuptools import Extension, setup

_NATIVE_MODE = os.environ.get("VN_ADDRESS_PARSER_NATIVE", "optional")
_VALID_NATIVE_MODES = {"disabled", "optional", "required"}
if _NATIVE_MODE not in _VALID_NATIVE_MODES:
    choices = ", ".join(sorted(_VALID_NATIVE_MODES))
    raise RuntimeError(f"VN_ADDRESS_PARSER_NATIVE must be one of: {choices}")

extensions: list[Extension] = []
if _NATIVE_MODE != "disabled":
    extensions = cythonize(
        [
            Extension(
                "address_parser._native_kernels",
                ["address_parser/_native_kernels.pyx"],
            )
        ],
        compiler_directives={
            "annotation_typing": True,
            "language_level": 3,
        },
    )
    for extension in extensions:
        extension.optional = _NATIVE_MODE == "optional"

setup(ext_modules=extensions)
