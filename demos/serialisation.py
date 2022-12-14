import tempfile
import subprocess
import os
import sys

from typed_python import SerializationContext


def callFunctionInFreshProcess(func, argTup, compilerCacheDir=None):
    """Return the value of a function evaluated on some arguments in a subprocess.

    We use this to test the semantics of anonymous functions and classes in a process
    that didn't create those obects.

    Args:
        func - the function object to call
        argTup - a tuple of arguments

    Returns:
        the result of the expression.
    """
    with tempfile.TemporaryDirectory() as tf:
        env = dict(os.environ)

        if compilerCacheDir:
            env["TP_COMPILER_CACHE"] = compilerCacheDir

        sc = SerializationContext()

        with open(os.path.join(tf, "input"), "wb") as f:
            f.write(sc.serialize((func, argTup)))

        try:
            subprocess.check_output(
                [
                    sys.executable,
                    "-u",
                    "-c",
                    "from typed_python import SerializationContext\n"
                    "import os\n"
                    "sc = SerializationContext()\n"
                    "if not os.path.exists('input'):\n"
                    "    raise ValueError()\n"
                    "with open('input', 'rb') as f:\n"
                    "    func, argTup = sc.deserialize(f.read())\n"
                    "with open('output', 'wb') as f:\n"
                    "    f.write(sc.serialize(func(*argTup)))\n",
                ],
                cwd=tf,
                env=env,
                stderr=subprocess.PIPE,
            )
        except subprocess.CalledProcessError as e:
            raise Exception(
                "Subprocess failed:\n\n"
                + e.stdout.decode("ASCII")
                + "\n\nerr=\n"
                + e.stderr.decode("ASCII")
            )

        with open(os.path.join(tf, "output"), "rb") as f:
            result = sc.deserialize(f.read())

        return result


class Cls:
    def f(self):
        return Cls


assert callFunctionInFreshProcess(Cls().f, ()) is Cls
