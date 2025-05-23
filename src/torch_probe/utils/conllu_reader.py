# src/torch_probe/utils/conllu_reader.py
from typing import List, Dict, Any, Iterator, Tuple

SentenceData = Dict[str, List[Any]] # Type alias for clarity

def _generate_sentence_lines(filepath: str) -> Iterator[List[str]]:
    """
    Reads a CoNLL-U file and yields lists of lines, each list representing one sentence.
    Skips comment lines.
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

def read_conllu_file(filepath: str) -> List[SentenceData]:
    """
    Reads a CoNLL-U file and parses sentences.

    Args:
        filepath: Path to the CoNLL-U file.

    Returns:
        A list of dictionaries. Each dictionary represents a sentence and contains:
        - 'tokens': List of word forms (str).
        - 'head_indices': List of 0-indexed integer head indices for each token.
                          Root token's head points to -1.
        - 'dep_rels': List of dependency relation labels (str). (Optional, good to have)
        - 'upos_tags': List of UPOS tags (str). (Optional, good to have)
        (Can be extended to include other CoNLL-U fields if needed)
    """
    parsed_sentences: List[SentenceData] = []

    for sentence_lines in _generate_sentence_lines(filepath):
        tokens: List[str] = []
        head_indices_str: List[str] = [] # Store as string first for root handling
        dep_rels: List[str] = []
        upos_tags: List[str] = []
        
        # Temporary storage for all columns of valid token lines
        token_line_columns: List[List[str]] = []

        for line in sentence_lines:
            parts = line.split('\t')
            if len(parts) < 8: # Ensure at least ID, FORM, ..., HEAD, DEPREL columns
                # print(f"Warning: Skipping malformed CoNLL-U line: {line}")
                continue
            
            token_id_str = parts[0]
            if '-' in token_id_str:  # Skip multi-word token lines like "1-2"
                continue
            if not token_id_str.isdigit(): # Skip non-token lines if any slipped through
                continue

            # If we reach here, it's a valid syntactic token line
            token_line_columns.append(parts)
            
            tokens.append(parts[1])          # FORM (Word form)
            upos_tags.append(parts[3])       # UPOS 
            head_indices_str.append(parts[6])# HEAD (1-indexed or 0 for root)
            dep_rels.append(parts[7])        # DEPREL

        if not tokens: # Skip if sentence had no valid token lines
            continue

        # Convert head indices to 0-indexed integers, root points to -1
        # CoNLL-U head '0' means root. We'll map it to -1.
        # Other heads are 1-indexed, so subtract 1.
        head_indices: List[int] = []
        for i, head_str in enumerate(head_indices_str):
            if not head_str.isdigit():
                # print(f"Warning: Non-integer head '{head_str}' for token '{tokens[i]}'. Defaulting to root (-1).")
                head_indices.append(-1) # Or handle error as preferred
                continue
            head_val = int(head_str)
            if head_val == 0: # Root in CoNLL-U
                head_indices.append(-1) 
            else:
                head_indices.append(head_val - 1) # Convert 1-indexed to 0-indexed

        parsed_sentences.append({
            'tokens': tokens,
            'head_indices': head_indices,
            'dep_rels': dep_rels,
            'upos_tags': upos_tags
            # Add other fields here if you parse more columns
        })
        
    return parsed_sentences

if __name__ == '__main__':
    # Example usage (create a dummy test.conllu file to test this)
    # You would replace 'test.conllu' with an actual file path
    # dummy_conllu_content = """
    # # sent_id = 1
    # # text = This is a test.
    # 1	This	this	PRON	DT	_	4	nsubj	_	_
    # 2	is	be	AUX	VBZ	_	4	cop	_	_
    # 3	a	a	DET	DT	_	4	det	_	_
    # 4	test	test	NOUN	NN	_	0	root	_	SpaceAfter=No
    # 5	.	.	PUNCT	.	_	4	punct	_	_

    # # sent_id = 2
    # # text = Multi-word tokens.
    # 1	Multi	multi	ADJ	JJ	_	3	amod	_	SpaceAfter=No
    # 2	-	-	PUNCT	HYPH	_	1	punct	_	SpaceAfter=No
    # 3	word	word	NOUN	NN	_	4	compound	_	_
    # 4	tokens	token	NOUN	NNS	_	0	root	_	SpaceAfter=No
    # 5	.	.	PUNCT	.	_	4	punct	_	_
    # """
    # with open("test.conllu", "w", encoding="utf-8") as f:
    #     f.write(dummy_conllu_content)

    # sentences = read_conllu_file("test.conllu")
    # for i, sent_data in enumerate(sentences):
    #     print(f"Sentence {i+1}:")
    #     print(f"  Tokens: {sent_data['tokens']}")
    #     print(f"  Heads (0-indexed, root=-1): {sent_data['head_indices']}")
    #     print(f"  UPOS: {sent_data['upos_tags']}")
    #     print(f"  DepRels: {sent_data['dep_rels']}")
    # os.remove("test.conllu") # Clean up dummy file
    pass # Keep __main__ block clean for module use