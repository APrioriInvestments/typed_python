#!/usr/bin/env python
"""
Runs pytest collection and format the results.

Passes any arguments to pytest.
"""
import os
import subprocess
import sys

import yaml

from typing import Optional


def run_pytest_collect(args) -> Optional[str]:
    """Run custom collection that includes markers.

    Normal collection only uses the test names. In order to get
    tests/ in the PYTHONPATH, and use our collector plugin,
    we must call using python -m pytest.
    """
    command = [
        sys.executable,
        "-m",
        "pytest",
        "-p",
        "typed_python.marker_collector_plugin",
        "--collect-tests-and-markers",
        "-q",
    ] + args
    try:
        output = subprocess.check_output(command, text=True, stderr=subprocess.STDOUT)
    except subprocess.CalledProcessError as e:
        output = e.output
        print(f"Error occurred: {e}")
        return None
    return output


def parse_to_yaml(pytest_output: str) -> str:
    """
    Convert the pytest output into yaml of the form:
    - unique_test_name:
        path: str
        labels: List[str]


    Assumes that if its a three-part tuple, with the first part being a path, then its
    test output. (Needed because some versions of pytest output arbitrary lines that don't
    follow this format.)
    """

    output = [line.strip().split("::") for line in pytest_output.split("\n")[:-4]]
    parsed_output = {}

    for line in output:
        try:
            path, name, markers = line
            if not os.path.exists(path):
                continue
            if markers:
                parsed_output[name] = {"path": path, "labels": markers.split("|")}
            else:
                parsed_output[name] = {"path": path}
        except ValueError:
            continue
    return yaml.dump(parsed_output)


def main():
    args = sys.argv[1:]
    output = run_pytest_collect(args)
    if output:
        parsed_output = parse_to_yaml(output)
        print(parsed_output)


if __name__ == "__main__":
    main()
