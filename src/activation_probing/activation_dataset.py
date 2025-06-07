from typing import TextIO

from jaxtyping import Float, Int
from torch import Tensor
from torch.utils.data import TensorDataset
import torch


class ActivationDataset(TensorDataset):
    def __init__(
            self,
            activations: Float[Tensor, "index d_model"],
            labels: Int[Tensor, "label"],
            meta_data: dict[str, str] = None
    ):
        super().__init__(activations, labels)
        self._meta_data = meta_data if meta_data is not None else {}

    @property
    def meta_data(self) -> dict[str, str]:
        """
        Returns the metadata of the dataset.
        :return: A dictionary containing metadata about the dataset.
        """
        return self._meta_data.copy()

    @property
    def class_labels(self) -> list[str]:
        """
        A map from class label index to class label string.
        """
        return self._meta_data.get("class_labels", [])

    @staticmethod
    def load_from_file(file: TextIO, device: str = "cuda") -> "ActivationDataset":
        """
        Loads an ActivationDataset from a file.
        :param file: A file-like object containing the dataset.
        :return: An instance of ActivationDataset.
        """
        # File is a jsonl file.
        import json
        activations = []
        labels = []
        meta_data = None
        file.seek(0)
        for line_index, line in enumerate(file):
            entry = json.loads(line)
            if line_index == 0:
                if "class_labels" not in entry:
                    raise ValueError(f"Expected first line to be metadata with class_labels, but got: {entry}")
                meta_data = entry
                continue
            if "activation" not in entry or "label" not in entry:
                raise ValueError(f"Expected activation entry in line {line_index + 1}, but got: {entry}")
            activations.append(entry["activation"])
            labels.append(entry["label"])
        return ActivationDataset(
            activations=torch.tensor(activations, dtype=torch.float32, device=device),
            labels=torch.tensor(labels, dtype=torch.int64, device=device),
            meta_data=meta_data,
        )
