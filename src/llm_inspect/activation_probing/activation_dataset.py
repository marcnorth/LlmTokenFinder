from typing import TextIO
import json
import torch
from jaxtyping import Float, Int
from torch import Tensor, nn
from torch.utils.data import TensorDataset, random_split, DataLoader
from .probe import ActivationProbe


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

    @property
    def activation_dim(self) -> int:
        """
        Returns the dimensionality of the activations.
        :return: The number of features in the activation vectors.
        """
        return self[0][0].shape[0] if len(self) > 0 else 0

    def train_probe(
            self,
            num_epochs: int = 10,
            batch_size: int = 32,
            learning_rate: float = 0.01,
            training_test_split: float = 0.8,
            device: str = None,
            save_to: BinaryIO | None = None,
    ) -> tuple[ActivationProbe, DataLoader, DataLoader, dict[str, list[float]] | None]:
        """
        Trains a single-layer probe on the dataset.
        :return: A tuple containing the trained probe model, the training and testing dataloaders and optionally the training history.
        """
        if device is None:
            device = self[0][0].device
        probe = ActivationProbe(
            self.activation_dim,
            len(self.class_labels),
            activation_dataset_meta_data=self.meta_data,
        ).to(device=device)
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(probe.parameters(), lr=learning_rate)
        training_dataloader, testing_dataloader = self._create_probe_training_dataloaders(training_test_split, batch_size)
        history = {
            "training_accuracy": [],
            "testing_accuracy": []
        }
        for epoch in range(num_epochs):
            self._train_probe_for_one_epoch(probe, training_dataloader, optimizer, criterion)
            history["training_accuracy"].append(self.evaluate_probe(probe, training_dataloader))
            history["testing_accuracy"].append(self.evaluate_probe(probe, testing_dataloader))
        probe.eval()
            probe.save_to_file(save_to, training_history=history)
        return (
            probe,
            training_dataloader,
            testing_dataloader,
            history
        )

    def _train_probe_for_one_epoch(
            self,
            probe: ActivationProbe,
            dataloader: DataLoader,
            optimizer: torch.optim.Optimizer,
            criterion: nn.Module
    ):
        probe.train()
        device = next(probe.parameters()).device
        for activations, labels in dataloader:
            activations, labels = activations.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = probe(activations)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

    def evaluate_probe(self, probe: ActivationProbe, dataloader: DataLoader) -> float:
        """
        :return: The accuracy of the probe on the dataset.
        """
        correct = 0
        total = 0
        probe.eval()
        device = next(probe.parameters()).device
        with torch.no_grad():
            for activations, labels in dataloader:
                activations, labels = activations.to(device), labels.to(device)
                outputs = probe(activations)
                _, predicted = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        return correct / total if total > 0 else 0.0

    def _create_probe_training_dataloaders(self, training_test_split: float, batch_size: int) -> tuple[DataLoader, DataLoader]:
        training_dataset_length = int(len(self) * training_test_split)
        training_dataset, testing_dataset = random_split(self, [training_dataset_length, len(self) - training_dataset_length])
        return (
            DataLoader(training_dataset, batch_size=batch_size, shuffle=True),
            DataLoader(testing_dataset, batch_size=batch_size, shuffle=False)
        )

    @staticmethod
    def load_from_file(file: TextIO, device: str = "cpu") -> "ActivationDataset":
        """
        Loads an ActivationDataset from a file.
        :param file: A file-like object containing the dataset.
        :return: An instance of ActivationDataset.
        """
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
