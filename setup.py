#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

from setuptools import setup
from setuptools.extension import Extension
from setuptools.command.build_ext import build_ext
import os

class BuildCpuExtension(build_ext):
    def build_extensions(self):
        for ext in self.extensions:
            ext.sources = [s for s in ext.sources if not s.endswith(('.cu', '.cuh'))]
        super().build_extensions()

setup(
    name="diff_gaussian_rasterization",
    packages=['diff_gaussian_rasterization'],
    ext_modules=[
        Extension(
            name="diff_gaussian_rasterization._C",
            sources=[
            "rasterize_points.cpp",
            "ext.cpp"],
            extra_compile_args={"cxx": ["-I" + os.path.join(os.path.dirname(os.path.abspath(__file__)), "third_party/glm/")]})
        ],
    cmdclass={
        'build_ext': BuildCpuExtension
    }
)
