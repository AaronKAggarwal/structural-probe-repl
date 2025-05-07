# scripts/create_conllu_sample.py
import os
from pathlib import Path

def create_sample(input_filepath, output_filepath, num_sentences):
    """
    Reads a CoNLL-U file and writes the first num_sentences to an output file.
    A sentence in CoNLL-U is a block of lines followed by a blank line.
    """
    count = 0
    print(f"Processing {input_filepath} -> {output_filepath} ({num_sentences} sentences)")
    with open(input_filepath, 'r', encoding='utf-8') as infile, \
         open(output_filepath, 'w', encoding='utf-8') as outfile:
        for line in infile:
            outfile.write(line)
            if line.strip() == "": # Blank line signifies end of a sentence
                count += 1
                if count >= num_sentences:
                    print(f"  Wrote {count} sentences.")
                    break
        if count < num_sentences:
            print(f"  Warning: Only found {count} sentences, requested {num_sentences}.")

def main():
    base_input_dir = Path("data_staging/ud_ewt_full")
    base_output_dir = Path("data_staging/my_ewt_sample_for_legacy_probe/example/data/en_ewt-ud-sample")
    
    base_output_dir.mkdir(parents=True, exist_ok=True)

    samples_config = {
        "train": {"file": "en_ewt-ud-train.conllu", "count": 100},
        "dev":   {"file": "en_ewt-ud-dev.conllu",   "count": 50},
        "test":  {"file": "en_ewt-ud-test.conllu",  "count": 50}
    }

    for split_type, config in samples_config.items():
        input_file = base_input_dir / config["file"]
        output_file = base_output_dir / config["file"] # Same filename, different dir
        create_sample(input_file, output_file, config["count"])

    print("\nCoNLL-U sampling complete.")
    print(f"Sample files created in: {base_output_dir.resolve()}")

if __name__ == "__main__":
    main()