import pytest
import warnings

# ignore deprecation warnings from librosa
def pytest_collection_modifyitems(items):
    for item in items:
        item.add_marker(pytest.mark.filterwarnings("ignore::DeprecationWarning"))