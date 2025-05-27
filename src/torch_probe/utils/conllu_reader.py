# src/torch_probe/utils/conllu_reader.py
from typing import List, Dict, Any, Iterator

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
MIN_EXPECTED_COLUMNS = max(COL_ID, COL_FORM, COL_UPOS, COL_XPOS, COL_HEAD, COL_DEPREL) + 1


SentenceData = Dict[str, List[Any]] # Type alias for clarity

def _generate_sentence_lines(filepath: str) -> Iterator[List[str]]:
    """
    Reads a CoNLL-U/CoNLL-X file and yields lists of lines, 
    each list representing one sentence. Skips comment lines.
    """
    with open(filepath, 'r', encoding='utf-8') as f:
        sentence_buffer: List[str] = []
        for line in f:
            line = line.strip()
            if not line:  # End of a sentence
                if sentence_buffer:
                    yield sentence_buffer
                    sentence_buffer = []
            elif line.startswith('#'):  # Skip comment lines
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
            parts = line.split('\t')
            
            if len(parts) < MIN_EXPECTED_COLUMNS:
                # print(f"Warning: Skipping malformed CoNLL line (not enough columns): {line} in file {filepath}")
                continue # Skip malformed lines
            
            token_id_str = parts[COL_ID]
            
            # Skip multi-word token lines (e.g., "1-2")
            if '-' in token_id_str:
                continue
            
            # Skip lines where ID is not a simple digit (might be comments or enhanced dep IDs like "1.1")
            if not token_id_str.isdigit():
                # print(f"Warning: Skipping non-standard ID line: {line} in file {filepath}")
                continue

            valid_token_data_parts.append(parts)

        if not valid_token_data_parts: # Skip if sentence had no valid token lines
            continue

        # Process the collected valid token lines for this sentence
        for parts in valid_token_data_parts:
            tokens.append(parts[COL_FORM])
            upos_tags.append(parts[COL_UPOS])
            xpos_tags.append(parts[COL_XPOS]) # PTB POS tags used by H&M for punctuation
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
                head_indices.append(head_val - 1) # Convert 1-indexed to 0-indexed

        parsed_sentences.append({
            'tokens': tokens,
            'head_indices': head_indices,
            'dep_rels': dep_rels,
            'upos_tags': upos_tags, # UPOS/CPOSTAG
            'xpos_tags': xpos_tags  # XPOS/POSTAG (PTB tags)
        })
        
    return parsed_sentences

if __name__ == '__main__':
    import os # Added for os.remove

    # Example usage with a CoNLL-X like structure (10 columns)
    # This example uses PTB-style POS tags in the XPOS column (index 4)
    dummy_conllx_content = """# sent_id = 1
# text = This is a test.
1	This	this	DT	DT	_	4	nsubj	_	_	_
2	is	be	VBZ	VBZ	_	4	cop	_	_	_
3	a	a	DT	DT	_	4	det	_	_	_
4	test	test	NN	NN	_	0	ROOT	_	_	_
5	.	.	.	.	_	4	punct	_	_	_

# sent_id = 2
# text = Old CoreNLP output.
1	Old	old	JJ	JJ	_	2	amod	_	_	_
2	CoreNLP	corenlp	NNP	NNP	_	3	compound	_	_	_
3	output	output	NN	NN	_	0	ROOT	_	_	_
4	.	.	.	.	_	3	punct	_	_	_
"""
    # Test with multi-word token (should be skipped)
    dummy_conllu_with_mwt = """
# sent_id = 3
# text = We ran.
1-2	We ran	_	_	_	_	_	_	_	_
1	We	we	PRON	PRP	_	2	nsubj	_	_
2	ran	run	VERB	VBD	_	0	root	_	SpaceAfter=No
3	.	.	PUNCT	.	_	2	punct	_	_
"""

    with open("test.conllx", "w", encoding="utf-8") as f:
        f.write(dummy_conllx_content)
        f.write(dummy_conllu_with_mwt)


    print("--- Testing with test.conllx ---")
    sentences = read_conll_file("test.conllx")
    for i, sent_data in enumerate(sentences):
        print(f"Sentence {i+1}:")
        print(f"  Tokens: {sent_data['tokens']}")
        print(f"  Heads (0-indexed, root=-1): {sent_data['head_indices']}")
        print(f"  UPOS/CPOSTAG: {sent_data['upos_tags']}")
        print(f"  XPOS/POSTAG: {sent_data['xpos_tags']}") # Important for H&M
        print(f"  DepRels: {sent_data['dep_rels']}")
    
    if os.path.exists("test.conllx"):
        os.remove("test.conllx")
    print("\n--- End of test.conllx ---")