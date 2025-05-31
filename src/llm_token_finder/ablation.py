import functools
import math
from jaxtyping import Float
from torch import Tensor
from transformer_lens import HookedTransformer, ActivationCache
from transformer_lens.hook_points import HookPoint
from llm_token_finder.activation_analyser import AttentionHead
from llm_token_finder.token_finder import Token


class AblationLlm:
    """
    A class that wraps a language model and allows for ablation of specific attention heads.
    """
    def __init__(self, llm: HookedTransformer):
        self.llm = llm

    def forward(self, text: str, heads_to_ablate: list[AttentionHead] = (), token_movement_to_ablate: list[tuple[int | Token, int | Token]] = ()) -> tuple[Float[Tensor, "batch_size sequence_length d_vocab"], ActivationCache]:
        """
        Forward pass through the model with optional ablation of specific attention heads.
        :param text: The input text to process.
        :param heads_to_ablate: A list of AttentionHead objects to ablate.
        :param token_movement_to_ablate: A list of tuples representing pairs of token positions to ablate movement between, ablate (from_position, to_position).
        :return: The output of the model and the activation cache.
        """
        self.llm.reset_hooks()
        if heads_to_ablate:
            self._add_attention_head_ablation_hooks(heads_to_ablate)
        if token_movement_to_ablate:
            self._add_token_movement_ablation_hooks(token_movement_to_ablate)
        input_token_ids = self.llm.tokenizer.encode(text, return_tensors="pt", add_special_tokens=True)
        output, cache = self.llm.run_with_cache(input_token_ids)
        return output, cache

    def _add_attention_head_ablation_hooks(self, heads_to_ablate: list[AttentionHead]) -> None:
        heads_by_layer = {}
        for head in heads_to_ablate:
            if head.layer not in heads_by_layer:
                heads_by_layer[head.layer] = []
            heads_by_layer[head.layer].append(head)
        for layer, layer_heads in heads_by_layer.items():
            hook_name = f"blocks.{layer}.attn.hook_v"
            hook_fn = functools.partial(self._ablate_attention_head, heads_to_ablate=layer_heads)
            self.llm.add_hook(hook_name, hook_fn)

    def _add_token_movement_ablation_hooks(self, token_movement_to_ablate: list[tuple[int | Token, int | Token]]) -> None:
        token_movement_to_ablate = [(
            from_token.index if isinstance(from_token, Token) else from_token,
            to_token.index if isinstance(to_token, Token) else to_token
        ) for from_token, to_token in token_movement_to_ablate]
        # Add the hook to every layer, as we want to ablate the movement between tokens for every attention head in every layer.
        for layer_index in range(self.llm.cfg.n_layers):
            hook_name = f"blocks.{layer_index}.attn.hook_attn_scores"
            hook_fn = functools.partial(self._ablate_token_movement, token_movement_to_ablate=token_movement_to_ablate)
            self.llm.add_hook(hook_name, hook_fn)

    @staticmethod
    def _ablate_attention_head(
            activation: Float[Tensor, "batch sequence_length n_head d_head"],
            hook: HookPoint,
            heads_to_ablate: list[AttentionHead]
    ) -> Float[Tensor, "batch sequence_length n_head d_head"]:
        """
        Ablates the specified attention heads by setting all of their activations to zero.
        """
        head_indexes = [head.head for head in heads_to_ablate]
        activation[:, :, head_indexes, :] = 0
        return activation

    @staticmethod
    def _ablate_token_movement(
            activation: Float[Tensor, "batch n_head sequence_length_q sequence_length_k"],
            hook: HookPoint,
            token_movement_to_ablate: list[tuple[int, int]]
    ) -> Float[Tensor, "batch n_head sequence_length_q sequence_length_k"]:
        """
        Ablates the movement between specified tokens by setting their activations to zero.
        """
        for from_position, to_position in token_movement_to_ablate:
            activation[:, :, to_position, from_position] = -math.inf
        return activation
