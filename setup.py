#   Copyright 2017-2019 Nativepython Authors
#
#   Licensed under the Apache License, Version 2.0 (the "License");
#   you may not use this file except in compliance with the License.
#   You may obtain a copy of the License at
#
#       http://www.apache.org/licenses/LICENSE-2.0
#
#   Unless required by applicable law or agreed to in writing, software
#   distributed under the License is distributed on an "AS IS" BASIS,
#   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#   See the License for the specific language governing permissions and
#   limitations under the License.

import pkg_resources
import setuptools

from distutils.command.build_ext import build_ext
from distutils.extension import Extension


class TypedPythonBuildExtension(build_ext):
    def run(self):
        self.include_dirs.append(
            pkg_resources.resource_filename('numpy', 'core/include')
        )

        build_ext.run(self)


extra_compile_args = [
    '-O2',
    '-fstack-protector-strong',
    '-Wformat',
    '-Wdate-time',
    '-Werror=format-security',
    '-std=c++14',
    '-Wno-sign-compare',
    '-Wno-narrowing',
    '-Wno-sign-compare',
    '-Wno-terminate',
    '-Wno-reorder',
    '-Wno-bool-compare',
    '-Wno-cpp'
]

ext_modules = [
    Extension(
        'typed_python._types',
        sources=[
            'typed_python/all.cpp',
        ],
        define_macros=[
            ("_FORTIFY_SOURCE", 2)
        ],
        include_dirs=['typed_python/lz4'],
        extra_compile_args=extra_compile_args
    )
]

INSTALL_REQUIRES = [line.strip() for line in open('requirements.txt')]

setuptools.setup(
    name='typed_python',
    version='0.2.6',
    description='opt-in strong typing at runtime for python, plus a compiler.',
    author='Braxton Mckee',
    author_email='braxton.mckee@gmail.com',
    url='https://github.com/aprioriinvestments/typed_python',
    packages=setuptools.find_packages(),
    cmdclass={'build_ext': TypedPythonBuildExtension},
    ext_modules=ext_modules,
    install_requires=INSTALL_REQUIRES,

    # https://pypi.org/classifiers/
    classifiers=[
        "License :: OSI Approved :: Apache Software License",
        "Programming Language :: Python :: 3 :: Only",
    ],
    python_requires='>=3.7',
    license="Apache Software License v2.0",

    include_package_data=True,
    data_files=[
        ("", ["requirements.txt"]),
    ],
    zip_safe=False
)
