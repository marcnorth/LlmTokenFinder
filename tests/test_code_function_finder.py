import unittest
from llm_token_finder import FunctionFinder


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