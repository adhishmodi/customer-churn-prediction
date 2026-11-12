import pytest

from notebook_helpers import assert_notebook_execution_state


def _cell(source, execution_count=None, outputs=None):
    return {
        "cell_type": "code",
        "source": source,
        "execution_count": execution_count,
        "outputs": [] if outputs is None else outputs,
    }


def test_accepts_fully_clean_notebook_with_empty_placeholder():
    assert_notebook_execution_state([_cell("x = 1"), _cell(""), _cell("print(x)")])


def test_accepts_fully_executed_notebook_with_and_without_outputs():
    cells = [
        _cell("x = 1", execution_count=1),
        _cell("print(x)", execution_count=2, outputs=[
            {"output_type": "stream", "text": "1"},
        ]),
    ]
    assert_notebook_execution_state(cells)


def test_rejects_partially_executed_notebook():
    cells = [_cell("x = 1", execution_count=1), _cell("y = 2")]
    with pytest.raises(AssertionError, match="fully clean or fully executed"):
        assert_notebook_execution_state(cells)


def test_rejects_saved_execution_error():
    cells = [_cell("raise RuntimeError()", execution_count=1, outputs=[
        {"output_type": "error", "ename": "RuntimeError", "evalue": "failed"},
    ])]
    with pytest.raises(AssertionError, match="fully clean or fully executed"):
        assert_notebook_execution_state(cells)


def test_rejects_outputs_without_execution_count():
    cells = [_cell("x = 1", outputs=[
        {"output_type": "stream", "text": "unexpected"},
    ])]
    with pytest.raises(AssertionError, match="fully clean or fully executed"):
        assert_notebook_execution_state(cells)


def test_rejects_non_positive_or_boolean_execution_counts():
    for count in (0, -1, True):
        with pytest.raises(AssertionError, match="fully clean or fully executed"):
            assert_notebook_execution_state([_cell("x = 1", execution_count=count)])
