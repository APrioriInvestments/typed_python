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


class NumpyBuildExtension(build_ext):
    """
    Used for when numpy headers are needed during build.
    Ensures that numpy will be installed before attempting
    to include any of its libraries
    """

    def run(self):
        self.include_dirs.append(
            pkg_resources.resource_filename('numpy', 'core/include'))
        build_ext.run(self)


extra_compile_args = [
    '-O2',
    '-fstack-protector-strong',
    '-Wformat',
    '-Wdate-time',
    '-Werror=format-security',
    '-std=c++14',
    '-Wno-sign-compare',
    '-Wno-narrowing'
]


ext_modules = [
    Extension(
        'object_database._types',
        sources=[
            'object_database/all.cpp',
        ],
        define_macros=[
            ("_FORTIFY_SOURCE", 2)
        ],
        extra_compile_args=extra_compile_args
    ),
    Extension(
        'typed_python._types',
        sources=[
            'typed_python/all.cpp',
        ],
        define_macros=[
            ("_FORTIFY_SOURCE", 2)
        ],
        extra_compile_args=extra_compile_args
    )
]


INSTALL_REQUIRES = [line.strip() for line in open('requirements.txt')]


with open('README.md', "r") as f:
    long_description = f.read()


setuptools.setup(
    name='nativepython',
    version='0.0.1dev2',
    description='Tools for generating machine code using python.',
    long_description=long_description,
    long_description_content_type='text/markdown',
    author='Braxton Mckee',
    author_email='braxton.mckee@gmail.com',
    url='https://github.com/aprioriinvestments/nativepython',
    packages=setuptools.find_packages(),
    cmdclass={'build_ext': NumpyBuildExtension},
    ext_modules=ext_modules,
    setup_requires=['numpy'],
    install_requires=INSTALL_REQUIRES,

    # https://pypi.org/classifiers/
    classifiers=[
        "License :: OSI Approved :: Apache Software License",
        "Programming Language :: Python :: 3 :: Only",
    ],

    license="Apache Software License v2.0",
    entry_points={
        'console_scripts': [
            'object_database_webtest=object_database.frontends.object_database_webtest:main',
            'object_database_service_manager=object_database.frontends.service_manager:main',
        ]
    },

    include_package_data=True,
    data_files=[
        ("", ["requirements.txt"]),
    ],
    zip_safe=False,
    python_requires='~=3.6',
)
