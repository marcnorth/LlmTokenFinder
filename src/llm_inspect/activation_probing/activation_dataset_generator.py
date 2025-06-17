import json
import tempfile
from dataclasses import dataclass
import datetime
from typing import TextIO, Callable, Generator
from transformer_lens import HookedTransformer
from ..activation_probing.activation_dataset import ActivationDataset
from ..ablation.ablation_llm import AblationLlm
from ..common.attention_head import AttentionHead
from ..token_finder.token_finder import Token


@dataclass(slots=True)
class ActivationGeneratorInput:
    text: str
    token_position: int
    label_class_index: int


class ActivationDatasetGenerator:
    def __init__(self,
        llm: HookedTransformer,
        input_generator: Callable[[], Generator[ActivationGeneratorInput, None, None]],
        class_labels: list[str],
        layer: int,
        head: int | None = None,
        meta_data: dict[str, str | int | None] = None
    ):
        """
        It is recommended to use the static factory methods to create instances of this class.
        :param llm: The language model to use for generating the dataset.
        :param input_generator: An iterator that yields input text for each forward pass. Each item should be a tuple of (input_text, token_position). Token_position can either be an int or a callable that will return an int given the corresponding input_text.
        :param class_labels: A list (map of int to str) of class labels for the dataset. This will be used to create metadata for the dataset.
        :param layer: The 0-indexed layer number from which to extract activations.
        :param head: The 0-indexed head number from which to extract activations. If set, activations from this head's output will be used. If not set, the residual stream of the layer will be used.
        :param meta_data: Additional metadata to save with the dataset.
        """
        self._llm = llm
        self._input_generator = input_generator
        self._class_labels = class_labels
        self._layer = layer
        self._head = head
        if meta_data is None:
            meta_data = {}
        self._extra_meta_data = meta_data

    def generate(
            self,
            heads_to_ablate: list[AttentionHead] = (),
            token_movement_to_ablate: list[tuple[int | Token, int | Token]] = ()
    ) -> ActivationDataset:
        with tempfile.TemporaryFile(mode="r+", encoding="utf-8") as file:
            self.generate_and_save_to(file, heads_to_ablate, token_movement_to_ablate)
            return ActivationDataset.load_from_file(file, device=self._llm.cfg.device)

    def generate_and_save_to(
            self,
            output_file: TextIO,
            heads_to_ablate: list[AttentionHead] = (),
            token_movement_to_ablate: list[tuple[int | Token, int | Token]] = ()
    ):
        """
        Generates the dataset and saves it to the specified output file.
        :param output_file: A file-like object to write the dataset to. Each line will be a JSON object containing the activation and label.
        :param heads_to_ablate: A list of AttentionHead objects to ablate.
        :param token_movement_to_ablate: A list of tuples representing pairs of token positions to ablate movement between, ablate (from_position, to_position).
        :return:
        """
        meta_data = self._extra_meta_data
        meta_data["heads_to_ablate"] = [str(head) for head in heads_to_ablate]
        meta_data["token_movement_to_ablate"] = [f"{from_token.index if isinstance(from_token, Token) else from_token}>{to_token.index if isinstance(to_token, Token) else to_token}" for from_token, to_token in token_movement_to_ablate]
        output_file.write(json.dumps(self._meta_data) + "\n")
        self._llm.reset_hooks()
        ablated_llm = AblationLlm(self._llm)
        for activation_input in self._input_generator():
            if len(self._class_labels) <= activation_input.label_class_index:
                raise ValueError(f"Label class index {activation_input.label_class_index} is out of bounds for class labels: {self._class_labels}")
            _, cache = ablated_llm.forward(
                activation_input.text,
                heads_to_ablate=heads_to_ablate,
                token_movement_to_ablate=token_movement_to_ablate,
                remove_batch_dim=True,
                names_filter=lambda name: name == f"blocks.{self._layer}.attn.hook_result" if self._head else f"blocks.{self._layer}.hook_resid_post"
            )
            if self._head is not None:
                # Extract the output of the specified attention head at a specific position (i.e. what is the attention head moving to that position)
                activations = cache["result", self._layer][activation_input.token_position][self._head]
            else:
                # Extract the residual stream of the specified layer at a specific position
                activations = cache["resid_post", self._layer][activation_input.token_position]
            output_file.write(json.dumps({
                "activation": activations.tolist(),
                "label": activation_input.label_class_index,
            }) + "\n")

    @property
    def _meta_data(self) -> dict[str, str | int | None]:
        return {
            "layer": self._layer,
            "head": self._head,
            "class_labels": self._class_labels,
            "llm": self._llm.cfg.model_name,
            "timestamp": datetime.datetime.now(tz=datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        } | self._extra_meta_data.copy()

    @staticmethod
    def create_residual_stream_generator(
        llm: HookedTransformer,
        input_generator: Callable[[], Generator[ActivationGeneratorInput, None, None]],
        class_labels: list[str],
        layer: int,
        meta_data: dict[str, str | int | None] = None
    ) -> "ActivationDatasetGenerator":
        """
        Creates a generator that yields the residual stream activations for the specified layer.
        :param llm: The language model to use for generating the dataset.
        :param input_generator: An iterator that yields input text for each forward pass. Each item should be a tuple of (input_text, token_position). Token_position can either be an int or a callable that will return an int given the corresponding input_text.
        :param class_labels: A list of class labels for the dataset. This will be used to create metadata for the dataset.
        :param layer: The 0-indexed layer number from which to extract residual stream activations.
        :param meta_data: Additional metadata to save with the dataset.
        :return: An instance of ActivationDatasetGenerator configured for residual stream activations.
        """
        return ActivationDatasetGenerator(
            llm,
            input_generator=input_generator,
            class_labels=class_labels,
            layer=layer,
            head=None,
            meta_data=meta_data
        )

    @staticmethod
    def create_attention_head_output_generator(
        llm: HookedTransformer,
        input_generator: Callable[[], Generator[ActivationGeneratorInput, None, None]],
        class_labels: list[str],
        head: AttentionHead,
        meta_data: dict[str, str | int | None] = None
    ) -> "ActivationDatasetGenerator":
        """
        Creates a generator that yields the output activations of the specified attention head.
        :param llm: The language model to use for generating the dataset.
        :param input_generator: An iterator that yields input text for each forward pass. Each item should be a tuple of (input_text, token_position). Token_position can either be an int or a callable that will return an int given the corresponding input_text.
        :param class_labels: A list of class labels for the dataset. This will be used to create metadata for the dataset.
        :param head: The attention head to extract output activations from.
        :param meta_data: Additional metadata to save with the dataset.
        :return: An instance of ActivationDatasetGenerator configured for the specified attention head.
        """
        llm.set_use_attn_result(True)
        return ActivationDatasetGenerator(
            llm,
            input_generator=input_generator,
            class_labels=class_labels,
            layer=head.layer,
            head=head.head,
            meta_data=meta_data
        )
