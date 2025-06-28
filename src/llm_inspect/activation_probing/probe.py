import datetime
from dataclasses import dataclass, field
from typing import Any, BinaryIO
import torch
from torch import nn


@dataclass
class TrainingHistory:
    training_accuracy: list[float] = field(default_factory=list)
    validation_accuracy: list[float] = field(default_factory=list)


class ActivationProbe(nn.Module):
    """
    A simple linear probe for activation data. Just a
    """

    _VERSION = 1

    def __init__(
            self,
            num_input_features: int,
            num_classes: int,
            hidden_size: int = 128,
            activation_dataset_meta_data: dict[str, Any] = None,
            meta_data: dict[str, Any] = None
    ):
        super().__init__()
        self.linear1 = nn.Linear(num_input_features, hidden_size)
        self.relu = nn.ReLU()
        self.linear2 = nn.Linear(hidden_size, num_classes)
        self._activation_dataset_meta_data = activation_dataset_meta_data
        self._meta_data = meta_data if meta_data is not None else {}

    def forward(self, x):
        x = self.linear1(x)
        x = self.relu(x)
        x = self.linear2(x)
        return x

    @property
    def training_history(self) -> TrainingHistory:
        if "training_history" not in self._meta_data:
            raise ValueError("No training history found in the probe metadata.")
        return self._meta_data["training_history"]

    @property
    def final_validation_accuracy(self) -> float:
        if len(self.training_history.validation_accuracy) == 0:
            raise ValueError("No validation accuracy found in the training history.")
        return self.training_history.validation_accuracy[-1]

    def add_training_history(self, training_history: TrainingHistory) -> None:
        if "training_history" not in self._meta_data:
            self._meta_data["training_history"] = training_history
        else:
            existing_history = self._meta_data["training_history"]
            existing_history.training_accuracy.extend(training_history.training_accuracy)
            existing_history.validation_accuracy.extend(training_history.validation_accuracy)

    def save_to_file(self, file: BinaryIO) -> None:
        meta_data = self._meta_data
        if self._activation_dataset_meta_data is not None:
            meta_data["activation_dataset"] = self._activation_dataset_meta_data
        meta_data["version"] = self._VERSION
        meta_data["num_input_features"] = self.linear1.in_features
        meta_data["num_classes"] = self.linear2.out_features
        meta_data["hidden_size"] = self.linear1.out_features
        meta_data["timestamp"] = datetime.datetime.now(tz=datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ") if "timestamp" not in meta_data else meta_data["timestamp"]
        data = {
            "meta_data": self._meta_data,
            "state_dict": self.state_dict()
        }
        torch.save(data, file)

    @staticmethod
    def load_from_file(file: BinaryIO, device: str = "cpu") -> "ActivationProbe":
        file.seek(0)
        with torch.serialization.safe_globals([TrainingHistory]):
            data = torch.load(file)
        meta_data = data["meta_data"]
        state_dict = data["state_dict"]
        if meta_data.get("version", 0) != ActivationProbe._VERSION:
            raise ValueError(f"Probe version mismatch: expected {ActivationProbe._VERSION}, got {meta_data.get('version', 0)}")
        probe = ActivationProbe(
            num_input_features=meta_data["num_input_features"],
            num_classes=meta_data["num_classes"],
            hidden_size=meta_data["hidden_size"],
            activation_dataset_meta_data=meta_data.get("activation_dataset"),
            meta_data=meta_data
        )
        probe.load_state_dict(state_dict)
        probe.to(device)
        return probe
