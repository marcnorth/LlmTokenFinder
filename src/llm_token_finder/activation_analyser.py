from attr import dataclass
from typing import Callable

import circuitsvis as cv
import torch
from jaxtyping import Float
from torch import Tensor
from transformer_lens import ActivationCache

#from library.token_display import TokenDisplay
#from library.token_display import html_for_pretty_colored_tokens, html_for_pretty_colored_tokens_multi
from llm_token_finder import TokenFinder


class AttentionHead:
    """
    Class representing an attention head in a transformer model.
    :param layer: The 0-indexed layer index of the attention head in the model.
    :param head: The 0-indexed head index of the attention head in the layer.
    """
    def __init__(self, layer: int, head: int):
        self.layer = layer
        self.head = head

    def __repr__(self):
        return f"{self.layer}.{self.head}"



class ActivationAnalyzer:
    """
    Class containing helper functions for analyzing activations
    and parsing input tokens
    """

    def __init__(self, tokens: list[str], cache: ActivationCache, space_token: str = "Ġ", new_line_token: str = "Ċ"):
        self.cache = cache.remove_batch_dim()
        self.token_finder = TokenFinder(tokens, space_token=space_token, new_line_token=new_line_token)
        #self.code_token_finder = CodeTokenFinder(tokens, space_token=space_token, new_line_token=new_line_token)
        self.space_token = space_token
        self.new_line_token = new_line_token

    def find_heads_matching_criteria(
        self,
        criteria: Callable[[Float[Tensor, "q v"]], bool]
    ) -> list[AttentionHead]:
        """
        Find the heads that match the given criteria for the given activations.
        If the input is batched, then a head will only match if the criteria is true for all batches.

        E.g. to find heads that always focus on the first token in the sequence:
        find_heads_matching_criteria(cache, lambda attention: torch.all(attention.argmax(-1) == 0))

        :param cache: Cache of activations from transformer lens
        :param criteria: Function that takes in an unbatched head attention pattern tensor of shape (q, v) and returns a boolean indicating whether the head matches the criteria
        :return: List of tuples of (layer index, head index) for heads that match the criteria
        """
        matching_heads: list[AttentionHead] = []
        layer_index = -1
        while True:
            layer_index += 1
            try:
                attention_patterns = self.cache["pattern", layer_index]
            except KeyError:
                # Assume we've reached the end of the layers
                break
            for head_index in range(attention_patterns.shape[0]):
                if criteria(attention_patterns[head_index]):
                    matching_heads.append(AttentionHead(layer_index, head_index))
        return matching_heads

    def colored_tokens_for_attention(
        self,
        layer: int,
        head: int,
    ) -> cv.tokens.RenderedHTML:
        """
        Generate colored tokens for the given attention head
        """
        token_display = TokenDisplay(space_token=self.space_token, new_line_token=self.new_line_token)
        attention = self.cache["pattern", layer][head]
        return token_display.html_for_pretty_colored_tokens_multi(self.token_finder.tokens, attention.transpose(-2, -1))

    def colored_tokens_for_attention_q(
        self,
        layer: int,
        head: int,
        q_index: int,
    ) -> cv.tokens.RenderedHTML:
        """
        Generate colored tokens for the given query token in the attention head
        """
        attention = self.cache["pattern", layer][head]
        return html_for_pretty_colored_tokens(self.token_finder.tokens, attention[q_index])

    def attention_pattern(self, layer: int, head: int) -> cv.attention.RenderedHTML:
        return cv.attention.attention_pattern(tokens=self.token_finder.tokens, attention=self.cache["pattern", layer][head])
