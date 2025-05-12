from typing import Callable, Self
from jaxtyping import Float
from torch import Tensor
from transformer_lens import ActivationCache, HookedTransformer
from transformers import PreTrainedTokenizer
from llm_token_finder import TokenFinder
from llm_token_finder.token_finder import Token


class AttentionHead:
    """
    Class representing an attention head in a transformer model.
    :param layer: The 0-indexed layer index of the attention head in the model.
    :param head: The 0-indexed head index of the attention head in the layer.
    """
    def __init__(self, layer: int, head: int):
        self.layer = layer
        self.head = head

    def __eq__(self, other):
        if not isinstance(other, AttentionHead):
            return NotImplemented
        return self.layer == other.layer and self.head == other.head

    def __hash__(self):
        return hash((self.layer, self.head))

    def __repr__(self):
        return f"{self.layer}.{self.head}"

    @staticmethod
    def intersection(heads: list[list["AttentionHead"]]) -> list["AttentionHead"]:
        """
        Find the intersection of a list of lists of AttentionHead objects.
        :param heads: A list of lists of AttentionHead objects.
        :return: A list of AttentionHead objects that are in all of the lists.
        """
        if not heads:
            return []
        intersection = set(heads[0])
        for head_list in heads[1:]:
            intersection &= set(head_list)
        return list(intersection)


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

    def find_heads_where_query_looks_at_value(self, query_token: Token | int, value_token: Token | int) -> list[AttentionHead]:
        """
        Find heads where the query token has the highest attention score on the value token.
        :param query_token: The token (or token index) to look for in the query
        :param value_token: The token (or token index) to look for in the value
        """
        return self.find_heads_where_query_looks_at_values(query_token, [value_token])

    def find_heads_where_query_looks_at_values(self, query_token: Token | int, value_tokens: list[Token | int]) -> list[AttentionHead]:
        """
        Find heads where the query token has the highest attention score on all value tokens.
        :param query_token: The token (or token index) to look for in the query
        :param value_tokens: The tokens (or token indices) to look for in the values
        """
        query_token_index = query_token.index if isinstance(query_token, Token) else query_token
        value_token_indices = [token.index if isinstance(token, Token) else token for token in value_tokens]
        def criteria(attention: Float[Tensor, "q v"]) -> bool:
            query_attention_scores = attention[query_token_index]
            most_looked_at_token_indexes = query_attention_scores.topk(len(value_tokens)).indices.tolist()
            return all(value_token_index in most_looked_at_token_indexes for value_token_index in value_token_indices)
        return self.find_heads_matching_criteria(criteria)

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
    def from_forward(llm: HookedTransformer, input: str) -> Self:
        tokens = llm.tokenizer.tokenize(input, add_special_tokens=True)
        token_ids = llm.tokenizer.encode(input, return_tensors="pt")
        _, activation_cache = llm.run_with_cache(token_ids)
        return ActivationAnalyzer(tokens, activation_cache)

    @staticmethod
    def create_from_tokenizer(tokenizer: PreTrainedTokenizer, tokens: list[str], cache: ActivationCache) -> "ActivationAnalyzer":
        space_special_character = tokenizer.tokenize(" hello")[0][0]
        new_line_special_character = tokenizer.tokenize("\nhello")[0][0]
        return ActivationAnalyzer(tokens, cache, space_token=space_special_character, new_line_token=new_line_special_character)