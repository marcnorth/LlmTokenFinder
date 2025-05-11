from dataclasses import dataclass
from transformers import PreTrainedTokenizer


class Token:
    """
    Class representing a token in a list of tokens.
    Is essentially just an index, but is aware of the list it belongs to
    """
    def __init__(self, index: int, context: list[str]):
        self.index = index
        self.context = context

    @property
    def value(self) -> str:
        return self.context[self.index]

    def __repr__(self):
        return f"Token(index={self.index}, value='{self.value}')"


class Scope:
    """
    Class representing a scope of indices in a list of tokens.
    Is essentially just a start and end index (inclusive), but is aware of the list it belongs to
    """
    def __init__(self, start: int, end: int, context: list[str]):
        self.start: int = start
        self.end: int = end if end is not None else len(context)
        self.context: list[str] = context

    def __len__(self) -> int:
        return self.end - self.start + 1

    @property
    def reversed_start(self) -> int:
        return len(self.context) - self.end - 1

    @property
    def reversed_end(self) -> int:
        return len(self.context) - self.start

    def find_first(self, token: str, allow_space_prefix=False) -> Token:
        return TokenFinder(self.context).find_first(token, self, allow_space_prefix)

    def find_last(self, token: str, allow_space_prefix=False) -> Token:
        return TokenFinder(self.context).find_last(token, self, allow_space_prefix)

    def find_all(self, token: str | list[str], allow_space_prefix=False) -> list[Token]:
        return TokenFinder(self.context).find_all(token, self, allow_space_prefix)

    def __contains__(self, index: int) -> bool:
        return self.start <= index <= self.end

    def __repr__(self):
        return f"Scope(start={self.start} '{self.context[self.start]}', end={self.end} '{self.context[self.end]}')"


class TokenFinder:
    """
    Class containing helper functions for finding tokens in a list of tokens
    """
    def __init__(self, tokens: list[str], space_token: str = "Ġ", new_line_token: str = "Ċ"):
        self.tokens: list[str] = tokens
        self.space_token: str = space_token
        self.new_line_token: str = new_line_token

    def find_first(self, token: str, scope: Scope = None, allow_space_prefix=False) -> Token:
        """
        Find the first instance of a token in the list of tokens
        :param token:
        :param scope:
        :param allow_space_prefix: Since tokenization often includes a space token before the actual token, allow either. E.g. if space_token is "Ġ" and token is "a", find first instance of either "Ġa" or "a"
        :return:
        """
        if allow_space_prefix:
            return self.find_first_of_any([f"{self.space_token}{token}", token], scope)
        index = self.tokens.index(
            token,
            scope.start if scope is not None else 0,
            scope.end+1 if scope is not None else len(self.tokens),
        )
        return Token(index, self.tokens)

    def find_first_of_any(self, tokens: list[str], scope: Scope = None, allow_space_prefix=False) -> Token:
        """
        Find the first instance of any of the tokens in the list of tokens
        :param tokens:
        :param scope:
        :param allow_space_prefix: Since tokenization often includes a space token before the actual token, allow either. E.g. if space_token is "Ġ" and token is "a", find first instance of either "Ġa" or "a"
        :return:
        """
        if allow_space_prefix:
            tokens = [f"{self.space_token}{token}" for token in tokens] + tokens
            return self.find_first_of_any(tokens, scope, allow_space_prefix=False)
        found_indices = []
        for token in tokens:
            try:
                found_indices.append(self.find_first(token, scope).index)
            except ValueError:
                pass
        if not found_indices:
            raise ValueError(f"None of the tokens '{tokens}' found")
        return Token(min(found_indices), self.tokens)

    def find_last(self, token: str, scope: Scope = None, allow_space_prefix=False) -> Token:
        if allow_space_prefix:
            return self.find_last_of_any([f"{self.space_token}{token}", token], scope)
        index = self.tokens[::-1].index(
            token,
            scope.reversed_start if scope is not None else 0,
            scope.reversed_end if scope is not None else len(self.tokens),
        )
        return Token(len(self.tokens) - index - 1, self.tokens)

    def find_last_of_any(self, tokens: list[str], scope: Scope = None, allow_space_prefix=False) -> Token:
        if allow_space_prefix:
            tokens = [f"{self.space_token}{token}" for token in tokens] + tokens
            return self.find_last_of_any(tokens, scope, allow_space_prefix=False)
        last_instances = []
        for token in tokens:
            try:
                last_instances.append(self.find_last(token, scope))
            except ValueError:
                pass
        if not last_instances:
            raise ValueError(f"Token(s) '{tokens}' not found")
        return max(last_instances, key=lambda t: t.index)

    def find_all(self, token: str, scope: Scope = None, allow_space_prefix=False) -> list[Token]:
        if allow_space_prefix:
            return self.find_all_of_any([f"{self.space_token}{token}", token], scope)
        return [Token(i, self.tokens) for i, t in enumerate(self.tokens) if t == token and (scope is None or i in scope)]

    def find_all_of_any(self, tokens: list[str], scope: Scope = None) -> list[Token]:
        instances = []
        for token in tokens:
            instances.extend(self.find_all(token, scope))
        instances.sort(key=lambda token: token.index)
        return instances

    @staticmethod
    def create_with_tokenizer(text: str, tokenizer: PreTrainedTokenizer) -> "TokenFinder":
        tokens = tokenizer.tokenize(text)
        space_special_character = tokenizer.tokenize(" hello")[0][0]
        new_line_special_character = tokenizer.tokenize("\nhello")[0][0]
        return TokenFinder(tokens, space_token=space_special_character, new_line_token=new_line_special_character)