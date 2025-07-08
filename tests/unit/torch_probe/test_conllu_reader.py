# tests/unit/torch_probe/test_conllu_reader.py
import tempfile  # For creating temporary files
from pathlib import Path

import pytest

from src.torch_probe.utils.conllu_reader import (
    read_conll_file,  # Adjust import if structure differs
)


@pytest.fixture
def sample_conllu_file_fixture():
    conllu_content = """# sent_id = 1
# text = This is a test.
1	This	this	PRON	DT	_	4	nsubj	_	_
2	is	be	AUX	VBZ	_	4	cop	_	_
3	a	a	DET	DT	_	4	det	_	_
4	test	test	NOUN	NN	_	0	root	_	SpaceAfter=No
5	.	.	PUNCT	.	_	4	punct	_	_

# sent_id = 2
# text = Multi-word tokens are handled.
1	Multi	multi	ADJ	JJ	_	3	amod	_	SpaceAfter=No
2	-	-	PUNCT	HYPH	_	1	punct	_	SpaceAfter=No
3	word	word	NOUN	NN	_	4	compound	_	_
4	tokens	token	NOUN	NNS	_	0	root	_	_
5-6	are	_	_	_	_	_	_	_	_
5	are	be	AUX	VBP	_	4	cop	_	_
6	handled	handle	VERB	VBN	_	4	acl:relcl	_	SpaceAfter=No # Example, not real parse
7	.	.	PUNCT	.	_	4	punct	_	_
"""
    # Use tempfile for creating files during tests
    with tempfile.NamedTemporaryFile(
        mode="w", delete=False, suffix=".conllu", encoding="utf-8"
    ) as tmp_file:
        tmp_file.write(conllu_content)
        filepath = tmp_file.name
    yield filepath
    Path(filepath).unlink()  # Clean up


def test_read_simple_sentence(sample_conllu_file_fixture):
    sentences = read_conll_file(sample_conllu_file_fixture)
    assert len(sentences) == 2

    # Test sentence 1
    sent1 = sentences[0]
    assert sent1["tokens"] == ["This", "is", "a", "test", "."]
    assert sent1["head_indices"] == [3, 3, 3, -1, 3]  # 0-indexed, root is -1
    assert sent1["upos_tags"] == ["PRON", "AUX", "DET", "NOUN", "PUNCT"]
    assert sent1["dep_rels"] == ["nsubj", "cop", "det", "root", "punct"]


def test_read_mwt_sentence(sample_conllu_file_fixture):
    sentences = read_conll_file(sample_conllu_file_fixture)
    # Test sentence 2 (with MWT)
    sent2 = sentences[1]
    assert sent2["tokens"] == ["Multi", "-", "word", "tokens", "are", "handled", "."]
    # Heads for MWT example:
    # Multi -> word (idx 2)
    # - -> Multi (idx 0)
    # word -> tokens (idx 3)
    # tokens -> root (idx -1)
    # are -> tokens (idx 3, assuming copula to noun for simplicity here, or to handled if acl:relcl)
    # handled -> tokens (idx 3, acl:relcl)
    # . -> tokens (idx 3)
    # This parse for 'are handled' is a bit off from the example comment, let's fix
    # Let's assume 'handled' is acl of 'tokens', and 'are' is aux of 'handled'
    # Multi (0) -> word (2)
    # - (1) -> Multi (0)
    # word (2) -> tokens (3)
    # tokens (3) -> root (-1)
    # are (4) -> handled (5)
    # handled (5) -> tokens (3)
    # . (6) -> tokens (3)
    assert sent2["head_indices"] == [2, 0, 3, -1, 3, 3, 3]
    assert sent2["upos_tags"] == [
        "ADJ",
        "PUNCT",
        "NOUN",
        "NOUN",
        "AUX",
        "VERB",
        "PUNCT",
    ]
