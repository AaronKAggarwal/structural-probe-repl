# src/torch_probe/utils/conllu_reader.py
from pathlib import Path
from typing import Any, Dict, Iterator, List

# Standard CoNLL column indices (0-indexed)
# These are common for CoNLL-U and CoNLL-X.
# Adjust if your specific CoNLL-X variant from older CoreNLP differs.
COL_ID = 0
COL_FORM = 1
COL_LEMMA = 2
COL_UPOS = 3  # Universal POS / Coarse POS in CoNLL-X
COL_XPOS = 4  # Language-specific POS / Fine-grained POS (e.g., PTB tags)
COL_FEATS = 5
COL_HEAD = 6
COL_DEPREL = 7
# COL_PHEAD = 8 (Usually not needed for basic parsing tasks)
# COL_PDEPREL = 9 (Usually not needed for basic parsing tasks)

# Minimum expected columns for a valid data line
MIN_EXPECTED_COLUMNS = (
    max(COL_ID, COL_FORM, COL_UPOS, COL_XPOS, COL_HEAD, COL_DEPREL) + 1
)


SentenceData = Dict[str, List[Any]]  # Type alias for clarity


def _generate_sentence_lines(filepath: str) -> Iterator[List[str]]:
    """
    Reads a CoNLL-U/CoNLL-X file and yields lists of lines,
    each list representing one sentence. Skips comment lines.
    """
    with open(filepath, "r", encoding="utf-8") as f:
        sentence_buffer: List[str] = []
        for line in f:
            line = line.strip()
            if not line:  # End of a sentence
                if sentence_buffer:
                    yield sentence_buffer
                    sentence_buffer = []
            elif line.startswith("#"):  # Skip comment lines
                continue
            else:
                sentence_buffer.append(line)
        if sentence_buffer:  # Yield the last sentence if file doesn't end with newline
            yield sentence_buffer


def read_conll_file(filepath: str) -> List[SentenceData]:
    """
    Reads a CoNLL-U or CoNLL-X file and parses sentences.
    This version is adapted to ensure XPOS tags are extracted for H&M alignment.

    Args:
        filepath: Path to the CoNLL-U/CoNLL-X file.

    Returns:
        A list of dictionaries. Each dictionary represents a sentence and contains:
        - 'tokens': List of word forms (str) from FORM column.
        - 'head_indices': List of 0-indexed integer head indices for each token.
                          Root token's head points to -1. (From HEAD column)
        - 'dep_rels': List of dependency relation labels (str) from DEPREL column.
        - 'upos_tags': List of UPOS/CPOSTAG tags (str) from UPOS/CPOSTAG column.
        - 'xpos_tags': List of XPOS/POSTAG tags (str) from XPOS/POSTAG column.
                       (Crucial for H&M punctuation filtering)
    """
    parsed_sentences: List[SentenceData] = []

    for sentence_lines in _generate_sentence_lines(filepath):
        tokens: List[str] = []
        head_indices_str: List[str] = []
        dep_rels: List[str] = []
        upos_tags: List[str] = []
        xpos_tags: List[str] = []

        # Store valid token lines to process after checking sentence integrity
        valid_token_data_parts: List[List[str]] = []

        for line_idx, line in enumerate(sentence_lines):
            parts = line.split("\t")

            if len(parts) < MIN_EXPECTED_COLUMNS:
                # print(f"Warning: Skipping malformed CoNLL line (not enough columns): {line} in file {filepath}")
                continue  # Skip malformed lines

            token_id_str = parts[COL_ID]

            # Skip multi-word token lines (e.g., "1-2")
            if "-" in token_id_str:
                continue

            # Skip lines where ID is not a simple digit (might be comments or enhanced dep IDs like "1.1")
            if not token_id_str.isdigit():
                # print(f"Warning: Skipping non-standard ID line: {line} in file {filepath}")
                continue

            valid_token_data_parts.append(parts)

        if not valid_token_data_parts:  # Skip if sentence had no valid token lines
            continue

        # Process the collected valid token lines for this sentence
        for parts in valid_token_data_parts:
            tokens.append(parts[COL_FORM])
            upos_tags.append(parts[COL_UPOS])
            xpos_tags.append(
                parts[COL_XPOS]
            )  # PTB POS tags used by H&M for punctuation
            head_indices_str.append(parts[COL_HEAD])
            dep_rels.append(parts[COL_DEPREL])

        # Convert head indices to 0-indexed integers, root points to -1
        # CoNLL standard: head '0' means root. We map it to -1.
        # Other heads are 1-indexed, so subtract 1.
        head_indices: List[int] = []
        for i, head_str in enumerate(head_indices_str):
            if not head_str.isdigit():
                # This case should be rare in clean CoNLL data from standard parsers.
                # If it occurs, treating as root or raising an error might be options.
                # H&M's task.py had logic for '_', let's assume for now parsers give digits.
                # print(f"Warning: Non-integer head '{head_str}' for token '{tokens[i]}'. Defaulting to root (-1). File: {filepath}")
                head_indices.append(-1)
                continue

            head_val = int(head_str)
            if head_val == 0:  # Root in CoNLL (0-indexed head)
                head_indices.append(-1)
            else:
                head_indices.append(head_val - 1)  # Convert 1-indexed to 0-indexed

        parsed_sentences.append(
            {
                "tokens": tokens,
                "head_indices": head_indices,
                "dep_rels": dep_rels,
                "upos_tags": upos_tags,  # UPOS/CPOSTAG
                "xpos_tags": xpos_tags,  # XPOS/POSTAG (PTB tags)
            }
        )

    return parsed_sentences


if __name__ == "__main__":
    print("--- Testing read_conll_file function ---")

    project_root = (
        Path(__file__).resolve().parent.parent.parent.parent
    )  # Assuming src/torch_probe/utils/conllu_reader.py

    test_conllx_file_path = (
        project_root
        / "data"
        / "ptb_stanford_dependencies_conllx"
        / "ptb3-wsj-dev.conllx"
    )

    num_sentences_to_print = 3

    if not test_conllx_file_path.exists():
        print(f"ERROR: Test CoNLL-X file not found at: {test_conllx_file_path}")
        print(
            "Please ensure the path is correct and the ptb_to_conllx.sh script has been run successfully."
        )
    else:
        print(f"Attempting to read and parse: {test_conllx_file_path}")
        try:
            parsed_sentences_data = read_conll_file(str(test_conllx_file_path))

            if not parsed_sentences_data:
                print("No sentences were parsed from the file.")
            else:
                print(f"Successfully parsed {len(parsed_sentences_data)} sentences.")
                print(
                    f"\n--- Displaying first {num_sentences_to_print} parsed sentences ---"
                )
                for i, sent_data in enumerate(
                    parsed_sentences_data[:num_sentences_to_print]
                ):
                    print(f"\nSentence {i + 1}:")
                    print(f"  Tokens:       {sent_data.get('tokens')}")
                    print(f"  Head Indices: {sent_data.get('head_indices')}")
                    print(f"  UPOS Tags:    {sent_data.get('upos_tags')}")
                    print(
                        f"  XPOS Tags:    {sent_data.get('xpos_tags')}"
                    )  # Verify this is populated
                    print(f"  Dep Relations:{sent_data.get('dep_rels')}")

                    # Optional: Check lengths consistency
                    if sent_data.get("tokens") and sent_data.get("head_indices"):
                        if len(sent_data["tokens"]) != len(sent_data["head_indices"]):
                            print(
                                f"  WARNING: Token count ({len(sent_data['tokens'])}) and head_indices count ({len(sent_data['head_indices'])}) mismatch!"
                            )
                    if sent_data.get("tokens") and sent_data.get("xpos_tags"):
                        if len(sent_data["tokens"]) != len(sent_data["xpos_tags"]):
                            print(
                                f"  WARNING: Token count ({len(sent_data['tokens'])}) and xpos_tags count ({len(sent_data['xpos_tags'])}) mismatch!"
                            )

        except Exception:
            print("An error occurred during parsing or processing:")
            import traceback

            traceback.print_exc()

    print("\n--- End of conllu_reader.py test ---")
