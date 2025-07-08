# scripts/convert_sample_conllu_to_raw.py
from pathlib import Path


def conllu_to_raw(conllu_path, raw_text_path):
    """
    Extracts sentences from a CoNLL-U file and writes them as raw text,
    one sentence per line, tokens space-separated.
    Assumes CoNLL-U tokens are in the second column (index 1).
    """
    print(f"Converting {conllu_path} to {raw_text_path}...")
    with (
        open(conllu_path, "r", encoding="utf-8") as infile,
        open(raw_text_path, "w", encoding="utf-8") as outfile,
    ):
        current_sentence_tokens = []
        for line in infile:
            line = line.strip()
            if not line:  # Blank line, end of sentence
                if current_sentence_tokens:
                    outfile.write(" ".join(current_sentence_tokens) + "\n")
                    current_sentence_tokens = []
            elif line.startswith("#"):  # Comment line
                continue
            else:
                parts = line.split("\t")
                if len(parts) > 1 and parts[0].isdigit():  # Token line
                    current_sentence_tokens.append(parts[1])
        # Write any remaining sentence if file doesn't end with blank line
        if current_sentence_tokens:
            outfile.write(" ".join(current_sentence_tokens) + "\n")
    print("Conversion done.")


def main():
    sample_data_base = Path(
        "data_staging/my_ewt_sample_for_legacy_probe/example/data/en_ewt-ud-sample"
    )
    files_to_convert = [
        "en_ewt-ud-train.conllu",
        "en_ewt-ud-dev.conllu",
        "en_ewt-ud-test.conllu",
    ]

    for conllu_filename in files_to_convert:
        conllu_filepath = sample_data_base / conllu_filename
        raw_text_filename = conllu_filename.replace(".conllu", ".txt")  # H&M convention
        raw_text_filepath = sample_data_base / raw_text_filename

        if conllu_filepath.exists():
            conllu_to_raw(conllu_filepath, raw_text_filepath)
        else:
            print(
                f"Warning: {conllu_filepath} not found, skipping raw text conversion."
            )

    print("\nRaw text conversion for samples complete.")
    print(f"Raw text files (.txt) created in: {sample_data_base.resolve()}")


if __name__ == "__main__":
    main()
