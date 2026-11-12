"""Shared assertions for notebook execution-state validation."""

def assert_notebook_execution_state(code_cells):
    """Accept a fully clean notebook or a fully executed notebook.

    Empty code cells are placeholders and must not contain outputs or execution
    counts. Executed cells may legitimately have no displayed output. Mixed
    execution states and saved execution errors are rejected.
    """
    assert code_cells, "Notebook must contain at least one code cell"
    executable_states = []
    for index, cell in enumerate(code_cells):
        source = "".join(cell.get("source", []))
        outputs = cell.get("outputs")
        assert isinstance(outputs, list), f"Code cell {index} outputs must be a list"
        assert all(isinstance(output, dict) for output in outputs), (
            f"Code cell {index} contains a malformed output"
        )
        execution_count = cell.get("execution_count")
        if not source.strip():
            assert execution_count is None, f"Empty code cell {index} has an execution count"
            assert outputs == [], f"Empty code cell {index} has saved outputs"
            continue
        executable_states.append((execution_count, outputs))

    assert executable_states, "Notebook must contain non-empty code cells"
    clean_notebook = all(
        count is None and outputs == []
        for count, outputs in executable_states
    )
    executed_notebook = all(
        type(count) is int
        and count > 0
        and all(output.get("output_type") != "error" for output in outputs)
        for count, outputs in executable_states
    )
    assert clean_notebook or executed_notebook, (
        "Notebook must be either fully clean or fully executed without errors"
    )
