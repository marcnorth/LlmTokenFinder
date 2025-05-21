import functools

from jaxtyping import Float
from torch import Tensor
from transformer_lens import HookedTransformer, ActivationCache
from transformer_lens.hook_points import HookPoint

from llm_token_finder.activation_analyser import AttentionHead


class AblationLlm:
    """
    A class that wraps a language model and allows for ablation of specific attention heads.
    """
    def __init__(self, llm: HookedTransformer):
        self.llm = llm

    def forward(self, text, heads_to_ablate: list[AttentionHead] = ()) -> tuple[Float[Tensor, "batch_size sequence_length d_vocab"], ActivationCache]:
        """
        Forward pass through the model with optional ablation of specific attention heads.
        :param text: The input text to process.
        :param heads_to_ablate: A list of AttentionHead objects to ablate.
        :return: The output of the model and the activation cache.
        """
        self.llm.reset_hooks()
        if heads_to_ablate:
            self._add_ablation_hooks(heads_to_ablate)
        output, cache = self.llm.run_with_cache(text)
        return output, cache

    def _add_ablation_hooks(self, heads_to_ablate: list[AttentionHead]) -> None:
        heads_by_layer = {}
        for head in heads_to_ablate:
            if head.layer not in heads_by_layer:
                heads_by_layer[head.layer] = []
            heads_by_layer[head.layer].append(head)
        for layer, layer_heads in heads_by_layer.items():
            hook_name = f"blocks.{layer}.attn.hook_v"
            hook_fn = functools.partial(self.ablate_attention_head, heads_to_ablate=layer_heads)
            self.llm.add_hook(hook_name, hook_fn)

    @staticmethod
    def ablate_attention_head(
            activation: Float[Tensor, "batch sequence_length n_head d_head"],
            hook: HookPoint,
            heads_to_ablate: list[AttentionHead]
    ) -> Float[Tensor, "batch sequence_length n_head d_head"]:
        head_indexes = [head.head for head in heads_to_ablate]
        activation[:, :, head_indexes, :] = 0
        return activation
