"""Independent unit tests for CoNLL‑U reader and gold‑label utilities.

These tests are deliberately self‑contained and do **not** rely on any data
or fixtures defined elsewhere in the project.  They synthesise minimal
CoNLL‑U snippets on the fly and write them to temporary files supplied by
pytest's ``tmp_path`` fixture.

Target modules under test
------------------------
* ``torch_probe.utils.conllu_reader.read_conllu_file``
* ``torch_probe.utils.gold_labels.calculate_tree_depths``
* ``torch_probe.utils.gold_labels.calculate_tree_distances``
"""

from __future__ import annotations

import pathlib
import sys
from textwrap import dedent

import numpy as np
import pytest

# ---------------------------------------------------------------------------
# Ensure that ``src`` is on the import path so that ``torch_probe`` can be
# imported regardless of how the tests are invoked (e.g. ``pytest -q`` from
# repo root or from the ``tests`` directory).
# ---------------------------------------------------------------------------
REPO_ROOT = pathlib.Path(__file__).resolve().parents[4]
SRC_PATH = REPO_ROOT / "src"
if SRC_PATH.is_dir() and str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))

# Now the actual imports we wish to test
from torch_probe.utils.conllu_reader import read_conllu_file  # noqa: E402
from torch_probe.utils.gold_labels import (  # noqa: E402
    calculate_tree_depths,
    calculate_tree_distances,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _write_conllu(tmp_path: pathlib.Path, content: str) -> pathlib.Path:
    """Write *content* to a new ``example.conllu`` in *tmp_path* and return it."""
    path = tmp_path / "example.conllu"
    path.write_text(dedent(content).lstrip(), encoding="utf-8")
    return path


# ---------------------------------------------------------------------------
# Tests for ``read_conllu_file``
# ---------------------------------------------------------------------------


def test_read_conllu_simple_sentence(tmp_path: pathlib.Path) -> None:
    """A well‑formed three‑token sentence is parsed correctly."""

    conllu = _write_conllu(
        tmp_path,
        """
        # sent_id = 1
        1\tI\t_\tPRON\t_\t_\t2\tnsubj\t_\t_
        2\tlike\t_\tVERB\t_\t_\t0\troot\t_\t_
        3\tapples\t_\tNOUN\t_\t_\t2\tobj\t_\t_
        """,
    )

    sentences = read_conllu_file(conllu)
    assert len(sentences) == 1

    sent = sentences[0]
    assert sent["tokens"] == ["I", "like", "apples"]
    # heads were 2,0,2   →   1,‑1,1 (0‑indexed, root = ‑1)
    assert sent["head_indices"] == [1, -1, 1]
    assert sent["upos_tags"] == ["PRON", "VERB", "NOUN"]
    assert sent["dep_rels"] == ["nsubj", "root", "obj"]


def test_read_conllu_sentence_boundaries(tmp_path: pathlib.Path) -> None:
    """Two sentences separated by blank line are returned separately."""

    conllu = _write_conllu(
        tmp_path,
        """
        1\tHi\t_\tINTJ\t_\t_\t0\troot\t_\t_

        1\tBye\t_\tINTJ\t_\t_\t0\troot\t_\t_
        """,
    )
    sentences = read_conllu_file(conllu)
    assert [s["tokens"] for s in sentences] == [["Hi"], ["Bye"]]


def test_read_conllu_skips_comments(tmp_path: pathlib.Path) -> None:
    conllu = _write_conllu(
        tmp_path,
        """
        # random comment that should be ignored
        1\tHello\t_\tINTJ\t_\t_\t0\troot\t_\t_
        """,
    )
    sentences = read_conllu_file(conllu)
    assert len(sentences) == 1 and sentences[0]["tokens"] == ["Hello"]


def test_read_conllu_handles_mwt(tmp_path: pathlib.Path) -> None:
    """Multi‑word‑token lines are ignored, but their parts are kept."""

    conllu = _write_conllu(
        tmp_path,
        """
        1-2\tLet's\t_\t_\t_\t_\t_\t_\t_\t_
        1\tLet\t_\tVERB\t_\t_\t2\taux\t_\t_
        2\t's\t_\tAUX\t_\t_\t0\troot\t_\t_
        """,
    )
    sent = read_conllu_file(conllu)[0]
    assert sent["tokens"] == ["Let", "'s"]
    # head indices were 2,0  →  1,‑1
    assert sent["head_indices"] == [1, -1]


def test_read_conllu_empty_file(tmp_path: pathlib.Path) -> None:
    path = tmp_path / "empty.conllu"
    path.touch()
    assert read_conllu_file(path) == []


def test_read_conllu_malformed_lines_are_ignored(tmp_path: pathlib.Path) -> None:
    conllu = _write_conllu(
        tmp_path,
        """
        1\tGood\t_\tADJ\t_\t_\t0\troot\t_\t_
        2\tdata\t_\tNOUN\t_\t_\t1\tobj\t_\t_
        3\tBAD_LINE_TOO_SHORT
        4\t!\t_\tPUNCT\t_\t_\t1\tpunct\t_\t_
        """,
    )
    sent = read_conllu_file(conllu)[0]
    # The malformed line should not appear; total tokens = 3
    assert sent["tokens"] == ["Good", "data", "!"]
    assert sent["head_indices"] == [-1, 0, 0]


# ---------------------------------------------------------------------------
# Tests for ``calculate_tree_depths``
# ---------------------------------------------------------------------------


def test_calculate_tree_depths_linear_chain() -> None:
    heads = [-1, 0, 1, 2]
    expected_depths = [0, 1, 2, 3]
    assert calculate_tree_depths(heads) == expected_depths


def test_calculate_tree_depths_star_graph() -> None:
    heads = [-1, 0, 0, 0]
    expected_depths = [0, 1, 1, 1]
    assert calculate_tree_depths(heads) == expected_depths


def test_calculate_tree_depths_complex_tree() -> None:
    heads = [-1, 0, 1, 1, 3]  # 0→1→{2,3} and 3→4
    expected_depths = [0, 1, 2, 2, 3]
    assert calculate_tree_depths(heads) == expected_depths


def test_calculate_tree_depths_single_token() -> None:
    assert calculate_tree_depths([-1]) == [0]


def test_calculate_tree_depths_empty_input() -> None:
    assert calculate_tree_depths([]) == []


def test_calculate_tree_depths_multiple_roots() -> None:
    """If multiple roots exist the implementation chooses the first one."""
    heads = [-1, -1, 0]
    depths = calculate_tree_depths(heads)
    # Token‑0 root depth 0; token‑2 attached to 0 ⇒ depth 1;
    # Token‑1 becomes unreachable and remains ‑1.
    assert depths == [0, -1, 1]


# ---------------------------------------------------------------------------
# Tests for ``calculate_tree_distances``
# ---------------------------------------------------------------------------


def test_calculate_tree_distances_linear_chain() -> None:
    heads = [-1, 0, 1, 2]
    expected = np.array([
        [0, 1, 2, 3],
        [1, 0, 1, 2],
        [2, 1, 0, 1],
        [3, 2, 1, 0],
    ])
    np.testing.assert_array_equal(calculate_tree_distances(heads), expected)


def test_calculate_tree_distances_star_graph() -> None:
    heads = [-1, 0, 0, 0]
    expected = np.array([
        [0, 1, 1, 1],
        [1, 0, 2, 2],
        [1, 2, 0, 2],
        [1, 2, 2, 0],
    ])
    np.testing.assert_array_equal(calculate_tree_distances(heads), expected)


def test_calculate_tree_distances_complex_tree() -> None:
    heads = [-1, 0, 1, 1, 3]
    expected = np.array([
        [0, 1, 2, 2, 3],
        [1, 0, 1, 1, 2],
        [2, 1, 0, 2, 3],
        [2, 1, 2, 0, 1],
        [3, 2, 3, 1, 0],
    ])
    np.testing.assert_array_equal(calculate_tree_distances(heads), expected)


def test_calculate_tree_distances_single_token() -> None:
    result = calculate_tree_distances([-1])
    np.testing.assert_array_equal(result, np.array([[0]]))


def test_calculate_tree_distances_empty_input() -> None:
    result = calculate_tree_distances([])
    assert result.shape == (0, 0)


def test_calculate_tree_distances_symmetry_and_diagonal() -> None:
    heads = [-1, 0, 0, 1, 1]
    dist = calculate_tree_distances(heads)
    # Symmetric
    np.testing.assert_array_equal(dist, dist.T)
    # Diagonal zeros
    assert np.all(np.diag(dist) == 0)
