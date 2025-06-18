import unittest
from transformers import AutoTokenizer

from llm_inspect import FunctionFinder, TokenRange, Token


class CodeScopeFinderTest(unittest.TestCase):

    def test_python_function(self):
        code_tokens = [
            "#", "Comment", "Ċ", # 0-2
            "def", "Ġother", "(", "Ġx", ":", "str", ")", "->", "Ġfloat", ":", "Ċ", # 3-13
            "ĠĠ", "pass", "Ċ", # 14-16
            "#", "Comment", "Ċ", # 17-19
            "def", "Ġfoo", "(", "Ġx", ":", "Ġint", ")", "->", "Ġstr", ":", "Ċ", # 20-30
            "ĠĠ", "Ġreturn", '"', "abc", '"', # 31-35
            "Ċ", # 36
            "#", "Comment", "Ċ" # 37-39
        ]
        foo_scope = FunctionFinder(code_tokens).find_function_scope("foo")
        self.assertEqual(20, foo_scope.start)
        self.assertEqual(36, foo_scope.end)
        self.assertEqual(21, foo_scope.function_name_token.index)
        self.assertEqual("Ġfoo", foo_scope.function_name_token.value)
        self.assertEqual(1, len(foo_scope.parameters))
        self.assertEqual(23, foo_scope.parameters[0].name.index)
        self.assertEqual("Ġx", foo_scope.parameters[0].name.value)
        self.assertEqual(25, foo_scope.parameters[0].type.index)
        self.assertEqual("Ġint", foo_scope.parameters[0].type.value)
        self.assertEqual(28, foo_scope.return_type_token.index)
        self.assertEqual("Ġstr", foo_scope.return_type_token.value)
        self.assertEqual(30, foo_scope.body_scope.start)
        self.assertEqual(36, foo_scope.body_scope.end)

    def test_python_multitoken_function_name(self):
        code_tokens = [
            "def", "Ġtwo", "Tokens", "(", "Ġx", ":", "str", ")", "->", "Ġfloat", ":", "Ċ", # 0-11
            "ĠĠ", "pass", "Ċ", # 12-14
            "#", "Comment", "Ċ", # 15-17
            "def", "Ġtwo", "(", "Ġx", ":", "Ġint", ")", "->", "Ġstr", ":", "Ċ", # 18-28
            "ĠĠ", "Ġreturn", '"', "abc", '"', # 29-33
            "Ċ", # 34
            "#", "Comment", "Ċ" # 35-37
        ]
        # Single-token function name
        single_token_scope = FunctionFinder(code_tokens).find_function_scope("two")
        self.assertEqual(18, single_token_scope.start)
        self.assertEqual(34, single_token_scope.end)
        self.assertIsInstance(single_token_scope.function_name_token, TokenRange)
        self.assertIsInstance(single_token_scope.function_name_token, Token)
        self.assertEqual(19, single_token_scope.function_name_token.index)
        self.assertEqual("Ġtwo", single_token_scope.function_name_token.value)
        # Two-token function name
        two_token_scope = FunctionFinder(code_tokens).find_function_scope("twoTokens")
        self.assertEqual(0, two_token_scope.start)
        self.assertEqual(14, two_token_scope.end)
        self.assertIsInstance(two_token_scope.function_name_token, TokenRange)
        self.assertNotIsInstance(two_token_scope.function_name_token, Token)
        self.assertEqual(1, two_token_scope.function_name_token.start)
        self.assertEqual(2, two_token_scope.function_name_token.end)
        self.assertEqual("ĠtwoTokens", two_token_scope.function_name_token.to_string())

    def test_python_multitoken_variable_name(self):
        code_tokens = [
            "def", "Ġfoo", "(", "Ġthree", "_", "tokens", ":", "str", ",", "one", ":", "str", ")", "->", "Ġfloat", ":", "Ċ",  # 0-16
            "ĠĠ", "pass", "Ċ" # 13-15
        ]
        foo_scope = FunctionFinder(code_tokens).find_function_scope("foo")
        # three_tokens
        self.assertEqual(2, len(foo_scope.parameters))
        self.assertIsInstance(foo_scope.parameters[0].name, TokenRange)
        self.assertNotIsInstance(foo_scope.parameters[0].name, Token)
        self.assertEqual(3, foo_scope.parameters[0].name.start)
        self.assertEqual(5, foo_scope.parameters[0].name.end)
        self.assertEqual("Ġthree_tokens", foo_scope.parameters[0].name.to_string())
        # one
        self.assertIsInstance(foo_scope.parameters[1].name, TokenRange)
        self.assertIsInstance(foo_scope.parameters[1].name, Token)
        self.assertEqual(9, foo_scope.parameters[1].name.index)
        self.assertEqual("one", foo_scope.parameters[1].name.value)

    def test_python_multitoken_variable_type(self):
        code_tokens = [
            "def", "Ġfoo", "(", "Ġx", ":", "List", "[", "str", "]", ",", "Ġy", ":", "str", ")", "->", "Ġfloat", ":", "Ċ",  # 0-17
            "ĠĠ", "pass", "Ċ" # 18-20
        ]
        foo_scope = FunctionFinder(code_tokens).find_function_scope("foo")
        # List[str]
        self.assertEqual(2, len(foo_scope.parameters))
        self.assertIsInstance(foo_scope.parameters[0].type, TokenRange)
        self.assertNotIsInstance(foo_scope.parameters[0].type, Token)
        self.assertEqual(5, foo_scope.parameters[0].type.start)
        self.assertEqual(8, foo_scope.parameters[0].type.end)
        self.assertEqual("List[str]", foo_scope.parameters[0].type.to_string())
        # str
        self.assertIsInstance(foo_scope.parameters[1].type, TokenRange)
        self.assertIsInstance(foo_scope.parameters[1].type, Token)
        self.assertEqual(12, foo_scope.parameters[1].type.index)
        self.assertEqual("str", foo_scope.parameters[1].type.value)

    # TODO: Multi-token function names in other languages

    def test_java_function(self):
        code_tokens = [
            "public", "Ġclass", "ĠFoo", "{", "Ċ", # 0-4
            "Ġ", "public", "Ġfloat", "Ġother", "(", "String", "Ġx", ")", "{", "Ċ", # 5-14
            "Ġ", "}", "Ċ", # 15-17
            "Ġ", "public", "ĠString", "Ġfoo", "(", "int", "Ġx", ")", "{", "Ċ", # 18-27
            "Ġ", "return", 'Ġ"', "abc", '"', "Ċ", # 28-33
            "}", "Ċ" # 34-35
        ]
        foo_scope = FunctionFinder(code_tokens).find_function_scope("foo")
        self.assertEqual(19, foo_scope.start)
        self.assertEqual(34, foo_scope.end)
        self.assertEqual(21, foo_scope.function_name_token.index)
        self.assertEqual("Ġfoo", foo_scope.function_name_token.value)
        self.assertEqual(1, len(foo_scope.parameters))
        self.assertEqual(24, foo_scope.parameters[0].name.index)
        self.assertEqual("Ġx", foo_scope.parameters[0].name.value)
        self.assertEqual(23, foo_scope.parameters[0].type.index)
        self.assertEqual("int", foo_scope.parameters[0].type.value)
        self.assertEqual(20, foo_scope.return_type_token.index)
        self.assertEqual("ĠString", foo_scope.return_type_token.value)
        self.assertEqual(27, foo_scope.body_scope.start)
        self.assertEqual(33, foo_scope.body_scope.end)

    def test_go_function(self):
        code_tokens = [
            "func", "Ġother", "(", "x", "Ġstring", ")", "Ġint", "{", "Ċ", # 0-8
            "Ġ", "return", "Ġ0", "Ċ", # 9-12
            "}", "Ċ", # 13-14
            "func", "Ġfoo", "(", "x", "Ġint", ")", "Ġstring", "{", "Ċ", # 15-23
            "Ġ", "return", '"', "abc", '"', "Ċ", # 24-29
            "}", "Ċ", # 30-32
            "#", "Comment", "Ċ" # 33-35
        ]
        foo_scope = FunctionFinder(code_tokens).find_function_scope("foo")
        self.assertEqual(15, foo_scope.start)
        self.assertEqual(30, foo_scope.end)
        self.assertEqual(16, foo_scope.function_name_token.index)
        self.assertEqual("Ġfoo", foo_scope.function_name_token.value)
        self.assertEqual(1, len(foo_scope.parameters))
        self.assertEqual(18, foo_scope.parameters[0].name.index)
        self.assertEqual("x", foo_scope.parameters[0].name.value)
        self.assertEqual(19, foo_scope.parameters[0].type.index)
        self.assertEqual("Ġint", foo_scope.parameters[0].type.value)
        self.assertEqual(21, foo_scope.return_type_token.index)
        self.assertEqual("Ġstring", foo_scope.return_type_token.value)
        self.assertEqual(23, foo_scope.body_scope.start)
        self.assertEqual(29, foo_scope.body_scope.end)

    def test_llama_tokenizer(self):
        tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b")
        code = """def foo_fn(x: List[str], y: str) -> float:
            other_func(x, y)
        """
        function_finder = FunctionFinder.create_from_tokenizer(code, tokenizer)
        foo_scope = function_finder.find_function_scope("foo_fn")
        self.assertIsNotNone(foo_scope)

    def test_gemma_tokenizer(self):
        tokenizer = AutoTokenizer.from_pretrained("google/gemma-2-2b")
        code = """def foo_fn(x: List[str], y: str) -> float:
            other_func(x, y)
        """
        function_finder = FunctionFinder.create_from_tokenizer(code, tokenizer)
        foo_scope = function_finder.find_function_scope("foo_fn")
        self.assertIsNotNone(foo_scope)
