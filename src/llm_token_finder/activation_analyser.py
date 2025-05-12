from typing import Callable
from jaxtyping import Float
from torch import Tensor
from transformer_lens import ActivationCache
from transformers import PreTrainedTokenizer
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

    @staticmethod
    def create_for_tokenizer(tokenizer: PreTrainedTokenizer, tokens: list[str], cache: ActivationCache) -> "ActivationAnalyzer":
        space_special_character = tokenizer.tokenize(" hello")[0][0]
        new_line_special_character = tokenizer.tokenize("\nhello")[0][0]
        return ActivationAnalyzer(tokens, cache, space_token=space_special_character, new_line_token=new_line_special_character)