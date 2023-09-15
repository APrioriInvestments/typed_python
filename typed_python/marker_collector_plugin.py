import pytest


def pytest_addoption(parser):
    parser.addoption(
        "--collect-tests-and-markers",
        action="store_true",
        help="Collect all tests and their markers",
    )


class MarkerCollectorPlugin:
    def pytest_collection_finish(self, session):
        if session.config.getoption("--collect-tests-and-markers"):
            items = session.items  # The filtered collection of test items
            for item in items:
                print(f"{item.nodeid}::{'|'.join(x.name for x in item.own_markers)}")
            pytest.exit("Done collecting tests and markers", returncode=0)


def pytest_configure(config):
    if config.getoption("--collect-tests-and-markers"):
        config.pluginmanager.register(MarkerCollectorPlugin())
