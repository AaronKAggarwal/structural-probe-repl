# scripts/convert_conllu_to_raw_generic.py
import sys
from pathlib import Path
import argparse # For command-line arguments

# --- Add src to path for direct execution (if conllu_reader is in src) ---
SCRIPT_DIR_CONV = Path(__file__).resolve().parent
PROJECT_ROOT_CONV = SCRIPT_DIR_CONV.parent
SRC_ROOT_CONV = PROJECT_ROOT_CONV / "src"
if str(SRC_ROOT_CONV) not in sys.path:
    sys.path.append(str(SRC_ROOT_CONV))
# --- End Path Addition ---
from torch_probe.utils.conllu_reader import read_conll_file # Use your modern reader

def conllu_to_raw_text_from_parsed(parsed_sentences_data, raw_text_path):
    """
    Writes sentences from pre-parsed CoNLL-U data to raw text.
    """
    print(f"Writing raw text to {raw_text_path}...")
    with open(raw_text_path, 'w', encoding='utf-8') as outfile:
        for sent_data in parsed_sentences_data:
            if sent_data['tokens']: # Ensure there are tokens
                outfile.write(" ".join(sent_data['tokens']) + "\n")
    print("Raw text writing done.")

def main():
    parser = argparse.ArgumentParser(description="Convert CoNLL-U/CoNLL-X file to raw text, one sentence per line.")
    parser.add_argument("input_conllu_filepath", type=str, help="Path to the input CoNLL-U/CoNLL-X file.")
    parser.add_argument("output_raw_text_filepath", type=str, help="Path to save the output raw text file.")
    args = parser.parse_args()

    input_path = Path(args.input_conllu_filepath)
    output_path = Path(args.output_raw_text_filepath)

    if not input_path.exists():
        print(f"ERROR: Input CoNLL file not found: {input_path}")
        sys.exit(1)
        
    output_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"Reading CoNLL data from: {input_path}")
    # Use your modern conllu_reader which handles MWTs correctly
    parsed_sentences = read_conll_file(str(input_path))
    
    if not parsed_sentences:
        print(f"No sentences parsed from {input_path}. Output file will be empty.")
    
    conllu_to_raw_text_from_parsed(parsed_sentences, output_path)
    print(f"Raw text saved to: {output_path}")

if __name__ == "__main__":
    main()