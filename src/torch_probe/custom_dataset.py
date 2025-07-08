import logging
from typing import Any, Dict, Optional

from torch_probe.dataset import ProbeDataset, load_elmo_embeddings_for_sentence

log = logging.getLogger(__name__)


class FilteredProbeDataset(ProbeDataset):
    """
    A version of ProbeDataset that skips sentences with token count mismatches.
    """

    def __init__(
        self,
        conllu_filepath: str,
        hdf5_filepath: str,
        embedding_layer_index: int,
        probe_task_type: str,
        embedding_dim: Optional[int] = None,
        skip_token_mismatch: bool = True,
    ):
        # Initialize base class but use our own __getitem__
        super().__init__(
            conllu_filepath,
            hdf5_filepath,
            embedding_layer_index,
            probe_task_type,
            embedding_dim,
        )

        self.skip_token_mismatch = skip_token_mismatch
        self.skipped_sentences = set()

        # Pre-check all sentences to find which ones to skip
        if self.skip_token_mismatch:
            log.info("Checking for token count mismatches...")
            for idx in range(len(self.parsed_sentences)):
                sentence_key = str(idx)
                try:
                    embeddings_np = load_elmo_embeddings_for_sentence(
                        self.hdf5_file_object, sentence_key, self.embedding_layer_index
                    )

                    if embeddings_np is None:
                        self.skipped_sentences.add(idx)
                        continue

                    num_tokens_conllu = len(self.parsed_sentences[idx]["tokens"])
                    num_tokens_elmo = embeddings_np.shape[0]

                    if num_tokens_elmo != num_tokens_conllu:
                        log.warning(
                            f"Token count mismatch for sentence key '{sentence_key}': "
                            f"CoNLL-U has {num_tokens_conllu} tokens, "
                            f"ELMo HDF5 has {num_tokens_elmo} embedding vectors. "
                            f"Skipping this sentence."
                        )
                        self.skipped_sentences.add(idx)
                except Exception as e:
                    log.warning(
                        f"Error processing sentence {sentence_key}: {e}. Skipping."
                    )
                    self.skipped_sentences.add(idx)

            log.info(
                f"Skipped {len(self.skipped_sentences)} sentences with token count mismatches."
            )

        # Create a valid indices mapping
        self.valid_indices = [
            i
            for i in range(len(self.parsed_sentences))
            if i not in self.skipped_sentences
        ]

    def __len__(self) -> int:
        return len(self.valid_indices)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        if not (0 <= idx < len(self.valid_indices)):
            raise IndexError(
                f"Index {idx} out of bounds for filtered dataset of length {len(self.valid_indices)}"
            )

        # Map to the original index
        original_idx = self.valid_indices[idx]

        # Get the original data using the parent class's logic
        return super().__getitem__(original_idx)
