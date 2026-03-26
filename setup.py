from setuptools import setup, Extension

ext_modules = [
    Extension(
        'typed_python._types',
        sources=['typed_python/all.cpp'],
        define_macros=[("_FORTIFY_SOURCE", "2")],
        include_dirs=['typed_python/lz4'],
        extra_compile_args=[
            '-O2',
            '-fstack-protector-strong',
            '-Wformat',
            '-Wdate-time',
            '-Werror=format-security',
            '-std=c++14',
            '-Wno-sign-compare',
            '-Wno-narrowing',
            '-Wno-terminate',
            '-Wno-reorder',
            '-Wno-bool-compare',
            '-Wno-cpp',
        ],
    )
]

setup(ext_modules=ext_modules)
