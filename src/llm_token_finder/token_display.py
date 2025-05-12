import circuitsvis as cv
import torch
from jaxtyping import Float
from torch import Tensor
from transformer_lens import ActivationCache
from transformers import PreTrainedTokenizer
from llm_token_finder import Scope
from llm_token_finder.activation_analyser import AttentionHead
from llm_token_finder.token_finder import Token


class TokenDisplayer:

    def __init__(self, space_token: str = "Ġ", new_line_token: str = "Ċ"):
        self.space_token = space_token
        self.new_line_token = new_line_token

    def html_for_token_with_context(self, token: Token, context_len: int = 5) -> cv.tokens.RenderedHTML:
        """
        Generate HTML for a token with context_len tokens on either side
        """
        return self.html_for_scope_with_context(Scope(token.index, token.index, token.context), context_len)

    def html_for_scope_with_context(self, scope: Scope, context_len: int = 5) -> cv.tokens.RenderedHTML:
        """
        Generate HTML for a range of tokens with context_len tokens on either side
        """
        start_of_context = max(0, scope.start - context_len)
        tokens_to_print = scope.context[start_of_context : scope.end + context_len + 1]
        values = torch.zeros(len(tokens_to_print))
        values[scope.start - start_of_context:- start_of_context + scope.end + 1] = 1
        return self.html_for_pretty_colored_tokens(tokens_to_print, values)

    def html_for_pretty_colored_tokens(self, tokens: list[str], values: Float[Tensor, "sequence"]) -> cv.tokens.RenderedHTML:
        """
        Wrapper around circuitsvis.tokens.colored_tokens that replaces special tokens
        """
        if values.ndim != 1:
            raise ValueError(f"Values must be 1D, given tensor has shape {values.shape} (did you forget to index into the batch dimension?)")
        tokens = [token.replace(self.new_line_token, "↵\n").replace(self.space_token, " ") for token in tokens]
        # TODO: Monospace font
        return cv.tokens.colored_tokens(tokens, values)

    def html_for_pretty_colored_tokens_multi(self, tokens: list[str], values: Float[Tensor, "q v"]) -> cv.tokens.RenderedHTML:
        """
        Wrapper around circuitsvis.tokens.colored_tokens_multi that replaces special tokens
        """
        if values.ndim != 2:
            raise ValueError(f"Values must be 2D, given tensor has shape {values.shape}")
        tokens = [token.replace(self.new_line_token, "↵\n").replace(self.space_token, " ") for token in tokens]
        return cv.tokens.colored_tokens_multi(tokens, values, tokens)

    def html_for_token_attention(self, tokens: list[str], activation_cache: ActivationCache, head: AttentionHead) -> cv.tokens.RenderedHTML:
        """
        Generate HTML for the attention pattern of a given head
        """
        attention = activation_cache["pattern", head.layer][head.head]
        return self.html_for_pretty_colored_tokens_multi(tokens, attention.transpose(-2, -1))

    def html_for_attention_pattern(selfself, tokens: list[str], activation_cache: ActivationCache, head: AttentionHead) -> cv.tokens.RenderedHTML:
        """:
        Generate HTML for the attention pattern of a given head
        """
        attention = activation_cache["pattern", head.layer][head.head]
        return cv.attention.attention_pattern(tokens, attention)

    @staticmethod
    def create_for_tokenizer(tokenizer: PreTrainedTokenizer) -> "TokenDisplayer":
        space_special_character = tokenizer.tokenize(" hello")[0][0]
        new_line_special_character = tokenizer.tokenize("\nhello")[0][0]
        return TokenDisplayer(space_token=space_special_character, new_line_token=new_line_special_character)