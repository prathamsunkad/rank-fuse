
import pytest

from rankfuse.baselines.grouped import clear_graph_cache


@pytest.fixture(autouse=True)
def clear_cuda_graph_cache():
    clear_graph_cache()
    yield
    clear_graph_cache()
