import torch
from torch.utils.data import Dataset


class GPTDatasetV1(Dataset):
    """Custom Dataset for training GPT models.
    Splits text into overlapping sequences of tokens.
    """
    def __init__(self, text: str, tokenizer, max_length: int, stride: int) -> None:
        self.input_ids = []
        self.output_ids = []

        tokens = tokenizer.encode(text)

        for i in range(0, len(tokens) - max_length, stride):
            input_seq = tokens[i : i + max_length]
            target_seq = tokens[i + 1 : i + max_length + 1]

            self.input_ids.append(input_seq)
            self.output_ids.append(target_seq)

    def __len__(self):
        """Return the number of samples in the dataset."""
        return len(self.input_ids)

    def __getitem__(self, index: int):
        """Retrieve a sample from the dataset.
        Args:
            index: Index of the sample to retrieve.
        Returns:
            Tuple of (input_ids, output_ids) as tensors.
        """
        return torch.Tensor(self.input_ids[index]), torch.Tensor(self.output_ids[index])
