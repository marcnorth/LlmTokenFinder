from typing import Callable
from jaxtyping import Float
from torch import Tensor
from transformer_lens import ActivationCache, HookedTransformer
from transformers import PreTrainedTokenizer
from ..token_finder.token_finder import TokenFinder, Token
from ..common.attention_head import AttentionHead


class AttentionHeadFinder:

    def __init__(self, tokens: list[str], cache: ActivationCache, space_token: str = "Ġ", new_line_token: str = "Ċ", bos_token: str | None = None):
        """
        It is recommended to use the static factory methods to create an instance of this class.
        """
        self.cache = cache.remove_batch_dim()
        self.token_finder = TokenFinder(tokens, space_token=space_token, new_line_token=new_line_token)
        self.space_token = space_token
        self.new_line_token = new_line_token
        self.bos_token = bos_token

    def find_heads_where_query_looks_at_value(
            self,
            query_token: Token | int,
            value_token: Token | int,
            ignore_bos: bool = False
    ) -> list[AttentionHead]:
        """
        Find heads where the query token has the highest attention score on the value token.
        :param query_token: The token (or token index) to look for in the query
        :param value_token: The token (or token index) to look for in the value
        :param ignore_bos: If True, ignore the attention score for the BOS token when looking for the value token. This is an option because attention heads often have high attention scores for the BOS token that we might want to ignore.
        """
        return self.find_heads_where_query_looks_at_values(query_token, [value_token], ignore_bos=ignore_bos)

    def find_heads_where_query_looks_at_values(
            self, query_token: Token | int,
            value_tokens: list[Token | int],
            ignore_bos: bool = False # TODO: Maybe change this to a threshold, so the other tokens must be above a certain percentage of the bos token if it is higher (could be implemented by dividing the bos attention score by the value before comparing)
    ) -> list[AttentionHead]:
        """
        Find heads where the query token has the highest attention score on all value tokens.
        :param query_token: The token (or token index) to look for in the query
        :param value_tokens: The tokens (or token indices) to look for in the values
        :param ignore_bos: If True, ignore the attention score for the BOS token when looking for the value tokens. This is an option because attention heads often have high attention scores for the BOS token that we might want to ignore.
        """
        query_token_index = query_token if isinstance(query_token, int) else query_token.index
        value_token_indices = [token if isinstance(token, int) else token.index for token in value_tokens]
        value_token_indices_inc_bos = value_token_indices.copy() + ([0])
        def criteria(attention: Float[Tensor, "q v"]) -> bool:
            query_attention_scores = attention[query_token_index]
            most_looked_at_token_indexes = query_attention_scores.topk(len(value_tokens)).indices.tolist()
            if all(value_token_index in most_looked_at_token_indexes for value_token_index in value_token_indices):
                return True
            # If we are ignoring the BOS token, then the BOS is allowed to be in the top tokens k+1 tokens
            if ignore_bos and self.bos_token is not None:
                most_looked_at_token_indexes_inc_bos = query_attention_scores.topk(len(value_tokens) + 1).indices.tolist()
                return all(value_token_index in most_looked_at_token_indexes_inc_bos for value_token_index in value_token_indices_inc_bos)
            return False

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
    def from_forward_pass(llm: HookedTransformer, input: str) -> "AttentionHeadFinder":
        tokens = llm.tokenizer.tokenize(input, add_special_tokens=True)
        token_ids = llm.tokenizer.encode(input, return_tensors="pt")
        _, activation_cache = llm.run_with_cache(token_ids)
        return AttentionHeadFinder.create_from_tokenizer(llm.tokenizer, tokens, activation_cache)

    @staticmethod
    def create_from_tokenizer(tokenizer: PreTrainedTokenizer, tokens: list[str], cache: ActivationCache) -> "AttentionHeadFinder":
        space_special_character = tokenizer.tokenize(" hello")[0][0]
        new_line_special_character = tokenizer.tokenize("\nhello")[0][0]
        # Some tokenizers have bos_token set, but don't actually use it in the tokenization process (e.g. gpt2), so we check for it here.
        bos_check = tokenizer.tokenize("hello", add_special_tokens=True)[0]
        bos_token = tokenizer.bos_token if tokenizer.bos_token is not None and bos_check == tokenizer.bos_token else None
        return AttentionHeadFinder(tokens, cache, space_token=space_special_character, new_line_token=new_line_special_character, bos_token=bos_token)