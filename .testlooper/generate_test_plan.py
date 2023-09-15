"""generate_test_plan.py

Currently returns a static YAML file.
"""
import argparse


TEST_PLAN = """
version: 1
environments:
    # linux docker container for running our pytest unit-tests
    linux-pytest:
        image:
            docker:
                dockerfile: .testlooper/environments/Dockerfile.linux-pytest
        variables:
            PYTHONPATH: ${REPO_ROOT}
        min-ram-gb: 10
builds:
    # skip
suites:
    group_one:
        kind: unit
        environment: linux-pytest
        dependencies:
        list-tests: |
            python .testlooper/collect_pytest_tests.py -m 'group_one'
        run-tests: |
            python .testlooper/run_pytest_tests.py -m 'group_one'
        timeout: 30

    group_two:
        kind: unit
        environment: linux-pytest
        dependencies:
        list-tests: |
            python .testlooper/collect_pytest_tests.py -m 'group_two'
        run-tests: |
            python .testlooper/run_pytest_tests.py  -m 'group_two'
        timeout: 30
"""

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate a test plan.")
    parser.add_argument("--output", type=str, default="test_plan.yaml")
    args = parser.parse_args()

    with open(args.output, "w") as f:
        f.write(TEST_PLAN)
