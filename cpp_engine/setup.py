from __future__ import annotations

import os
import sys
from pathlib import Path

from pybind11.setup_helpers import Pybind11Extension, build_ext
from setuptools import setup

ROOT = Path(__file__).resolve().parent
use_openmp = os.environ.get("TURBOFAN_CPP_NO_OPENMP") != "1"

if sys.platform == "win32" and os.environ.get("TURBOFAN_CPP_MINGW") != "1":
    compile_args = ["/O2"]
    if use_openmp:
        compile_args.append("/openmp")
    link_args: list[str] = []
else:
    compile_args = ["-O3"]
    link_args = []
    if use_openmp:
        compile_args.append("-fopenmp")
        link_args.append("-fopenmp")

setup(
    name="turbofan-cpp-engine",
    version="1.0.0",
    description="Optional C++17 PCA and K-Means kernels for C-MAPSS analysis",
    ext_modules=[
        Pybind11Extension(
            "turbofan_cpp",
            sources=[
                str(ROOT / "src" / "bindings.cpp"),
                str(ROOT / "src" / "pca.cpp"),
                str(ROOT / "src" / "kmeans.cpp"),
            ],
            include_dirs=[str(ROOT / "include")],
            cxx_std=17,
            extra_compile_args=compile_args,
            extra_link_args=link_args,
        )
    ],
    cmdclass={"build_ext": build_ext},
    zip_safe=False,
    python_requires=">=3.12",
)
