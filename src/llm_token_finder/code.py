from dataclasses import dataclass
from transformers import PreTrainedTokenizer
from llm_token_finder.token_finder import Token, Scope, TokenFinder


@dataclass
class FunctionParameter:
    name: Token
    type: Token


class CodeFunction:
    """
    Class representing the scope of a function in a list of tokens, along with the important tokens of the function
    """

    def __init__(
            self,
            function_scope: Scope,
            body_scope: Scope,
            function_name_token: Token,
            parameters: list[FunctionParameter],
            return_type_token: Token
    ):
        self.function_scope: Scope = function_scope
        self.body_scope: Scope = body_scope
        self.function_name_token: Token = function_name_token
        self.parameters: list[FunctionParameter] = parameters
        self.return_type_token: Token = return_type_token

    @property
    def start(self) -> int:
        return self.function_scope.start

    @property
    def end(self) -> int:
        return self.function_scope.end


class FunctionFinder:
    """
    Class for finding the scope and relevant tokens of a function in a list of tokens in different programming languages.
    Currently supports Python, Go, and Java.
    This class was made for a specific experiment and probably isn't robust enough for general use.
    Assumes function/variable names and types are single tokens.
    """

    def __init__(self, tokens: list[str], space_token: str = "Ġ", new_line_token: str = "Ċ"):
        self.tokens = tokens
        self.space_token = space_token
        self.new_line_token = new_line_token
        self._language = self._detect_language()
        self._token_finder = TokenFinder(self.tokens, space_token=self.space_token)

    def _detect_language(self) -> str:
        """
        Detect the programming language of the tokens, supporting Python, Go, and Java.
        This is pretty crude and just looks for some common keywords. Works for the specific experiment this was made for,
        but would need to be improved for general use.
        """
        if "def" in self.tokens:
            return "python"
        if "func" in self.tokens:
            return "go"
        if "double" in self.tokens or "int" in self.tokens or "float" in self.tokens or "void" in self.tokens or "String" in self.tokens or "boolean" in self.tokens:
            return "java"
        raise ValueError("Could not detect programming language")

    @property
    def language(self) -> str:
        return self._language

    def find_function_scope(self, function_name: str) -> CodeFunction:
        """
        Find the start and end of a function scope given the function name
        """
        match self._language:
            case "python":
                return self._find_python_function_scope(function_name)
            case "go":
                return self._find_go_function_scope(function_name)
            case "java":
                return self._find_java_function_scope(function_name)

    def _find_python_function_scope(self, function_name: str) -> CodeFunction:
        """
        Find the start and end of a python function scope given the function name
        """
        function_name_token = self._token_finder.find_first(function_name, allow_space_prefix=True)
        if self.tokens[function_name_token.index - 1] != "def" and self.tokens[function_name_token.index - 1] != f"{self.space_token}def":
            print(f"Found function name '{function_name}' at index {function_name_token.index}")
            for i in range(10):
                print(f"{i}: {self.tokens[i]}")
            raise ValueError(f"Function name '{function_name}' is not preceded by 'def': '{self.tokens[function_name_token.index - 1]}'")
        # Find parameters
        parameters = []
        next_closing_bracket = self._token_finder.find_first(")", Scope(function_name_token.index, None, self.tokens))
        colons = self._token_finder.find_all(":", Scope(function_name_token.index, next_closing_bracket.index, self.tokens))
        for colon in colons:
            parameters.append(FunctionParameter(
                name=Token(colon.index - 1, self.tokens),
                type=Token(colon.index + 1, self.tokens)
            ))
        # Find return type
        next_colon = self._token_finder.find_first(":", Scope(next_closing_bracket.index, None, self.tokens))
        return_type_token = Token(next_colon.index - 1, self.tokens)
        # Find next line with no indentation
        try:
            next_no_indentation = len(self.tokens) - 1
            for i, token in enumerate(self.tokens[function_name_token.index:]):
                if token == self.new_line_token and not self.tokens[function_name_token.index + i + 1].startswith(self.space_token) and not self.tokens[function_name_token.index + i + 1].startswith(" "):
                    next_no_indentation = function_name_token.index + i
                    break
        except IndexError:
            next_no_indentation = len(self.tokens)
        if next_no_indentation < function_name_token.index:
            raise ValueError(f"next_no_indentation is less than function_name_token.index: {next_no_indentation} < {function_name_token.index}")
        scope = Scope(function_name_token.index - 1, next_no_indentation, self.tokens)
        return CodeFunction(
            scope,
            Scope(return_type_token.index + 2, scope.end, self.tokens),
            function_name_token,
            parameters,
            return_type_token
        )

    def _find_go_function_scope(self, function_name: str) -> CodeFunction:
        """
        Find the start and end of a Go function scope given the function name
        """
        try:
            function_name_token = self._token_finder.find_first(function_name)
            if self.tokens[function_name_token.index - 1] != "func":
                raise ValueError(f"Function name '{function_name}' is not preceded by 'func': {self.tokens[function_name_token.index - 1]}")
        except ValueError as e:
            if not function_name.startswith(self.space_token):
                return self.find_function_scope(self.space_token + function_name)
            raise e
        # Use brackets to find scope
        bracket_count = None
        scope = None
        for i, token in enumerate(self.tokens[function_name_token.index:]):
            if token == "{" or token == f"{self.space_token}{{":
                bracket_count = 1 if bracket_count is None else bracket_count + 1
            elif token == "}" or token == f"{self.space_token}}}":
                bracket_count -= 1
            if bracket_count == 0:
                scope = Scope(function_name_token.index - 1, function_name_token.index + i, self.tokens)
                break
        if scope is None:
            raise ValueError(f"Could not find scope for function {function_name}")
        # Find parameters
        parameters = []
        opening_bracket = self._token_finder.find_first("(", Scope(function_name_token.index, None, self.tokens))
        next_closing_bracket = self._token_finder.find_first(")", Scope(function_name_token.index, None, self.tokens))
        parameters_scope = Scope(opening_bracket.index, next_closing_bracket.index, self.tokens)
        if len(parameters_scope) > 2:
            parameters.append(FunctionParameter(
                name=Token(parameters_scope.start + 1, self.tokens),
                type=Token(parameters_scope.start + 2, self.tokens)
            ))
            commas = self._token_finder.find_all(",", parameters_scope)
            for comma in commas:
                parameters.append(FunctionParameter(
                    name=Token(comma.index + 1, self.tokens),
                    type=Token(comma.index + 2, self.tokens)
                ))
        body_open_token = self._token_finder.find_first("{", scope, allow_space_prefix=True)
        return CodeFunction(
            scope,
            Scope(body_open_token.index + 1, scope.end - 1, self.tokens),
            function_name_token,
            parameters,
            Token(next_closing_bracket.index + 1, self.tokens)
        )

    def _find_java_function_scope(self, function_name: str) -> CodeFunction:
        """
        Find the start and end of a java function scope given the function name
        """
        try:
            function_name_token = self._token_finder.find_first(function_name)
            if self.tokens[function_name_token.index - 1] not in [f"{self.space_token}double", f"{self.space_token}int", f"{self.space_token}float", f"{self.space_token}void", f"{self.space_token}String", f"{self.space_token}boolean"]:
                raise ValueError(f"Function name '{function_name}' is not preceded by a return type: {self.tokens[function_name_token.index - 1]}")
        except ValueError as e:
            if not function_name.startswith(self.space_token):
                return self.find_function_scope(f"{self.space_token}{function_name}")
            raise e
        # Use brackets to find scope
        bracket_count = None
        scope = None
        for i, token in enumerate(self.tokens[function_name_token.index:]):
            if token == "{" or token == f"{self.space_token}{{":
                bracket_count = 1 if bracket_count is None else bracket_count + 1
            elif token == "}" or token == f"{self.space_token}}}":
                bracket_count -= 1
            if bracket_count == 0:
                scope = Scope(function_name_token.index - 2, function_name_token.index + i, self.tokens)
                break
        if scope is None:
            raise ValueError(f"Could not find scope for function {function_name}")
        # Find parameters
        parameters = []
        opening_bracket = self._token_finder.find_first("(", Scope(function_name_token.index, None, self.tokens))
        next_closing_bracket = self._token_finder.find_first(")", Scope(function_name_token.index, None, self.tokens))
        parameters_scope = Scope(opening_bracket.index, next_closing_bracket.index, self.tokens)
        if len(parameters_scope) > 2:
            parameters.append(FunctionParameter(
                name=Token(parameters_scope.start + 2, self.tokens),
                type=Token(parameters_scope.start + 1, self.tokens)
            ))
            commas = self._token_finder.find_all(",", parameters_scope)
            for comma in commas:
                parameters.append(FunctionParameter(
                    name=Token(comma.index + 2, self.tokens),
                    type=Token(comma.index +1, self.tokens)
                ))
        body_open_token = self._token_finder.find_first("{", scope, allow_space_prefix=True)
        return CodeFunction(
            scope,
            Scope(body_open_token.index + 1, scope.end - 1, self.tokens),
            function_name_token,
            parameters,
            Token(function_name_token.index - 1, self.tokens)
        )

    @staticmethod
    def create_for_tokenizer(tokenizer: PreTrainedTokenizer, tokens) -> "FunctionFinder":
        space_special_character = tokenizer.tokenize(" hello")[0][0]
        new_line_special_character = tokenizer.tokenize("\nhello")[0][0]
        return FunctionFinder(tokens, space_token=space_special_character, new_line_token=new_line_special_character)