import unittest
from llm_token_finder import TokenFinder, Scope


class TokenFinderTest(unittest.TestCase):

    def setUp(self):
        self.tokens = [
            "Ġa", # 0
            "Ġb", # 1
            "Ġc", # 2
            "Ġa", # 3
            "Ġb", # 4
            "Ġc", # 5
            "a", # 6
            "b", # 7
            "c", # 8
            "a", # 9
            "b", # 10
            "c" # 11
        ]
        self.token_finder = TokenFinder(self.tokens, space_token="Ġ")

    def test_find_first(self):
        self.assertEqual(7, self.token_finder.find_first("b").index)

    def test_find_first_allow_space(self):
        self.assertEqual(1, self.token_finder.find_first("b", allow_space_prefix=True).index)

    def test_find_first_in_scope(self):
        scope = Scope(8, 10, self.tokens)
        self.assertEqual(10, self.token_finder.find_first("b", scope).index)

    def test_find_first_in_scope_allow_space(self):
        scope = Scope(3, 10, self.tokens)
        self.assertEqual(4, self.token_finder.find_first("b", scope, allow_space_prefix=True).index)

    def test_find_first_of_any(self):
        self.assertEqual(7, self.token_finder.find_first_of_any(["c", "b"]).index)

    def test_find_first_of_any_allow_space(self):
        self.assertEqual(1, self.token_finder.find_first_of_any(["c", "b"], allow_space_prefix=True).index)

    def test_find_first_of_any_in_scope(self):
        scope = Scope(6, 10, self.tokens)
        self.assertEqual(7, self.token_finder.find_first_of_any(["c", "b"], scope).index)

    def test_find_first_of_any_in_scope_allow_space(self):
        scope = Scope(3, 10, self.tokens)
        self.assertEqual(4, self.token_finder.find_first_of_any(["c", "b"], scope, allow_space_prefix=True).index)

    def test_find_last(self):
        self.assertEqual(10, self.token_finder.find_last("b").index)

    def test_find_last_allow_space(self):
        self.assertEqual(10, self.token_finder.find_last("b", allow_space_prefix=True).index)

    def test_find_last_in_scope(self):
        scope = Scope(0, 8, self.tokens)
        self.assertEqual(7, self.token_finder.find_last("b", scope).index)

    def test_find_last_in_scope_allow_space(self):
        scope = Scope(0, 8, self.tokens)
        self.assertEqual(7, self.token_finder.find_last("b", scope, allow_space_prefix=True).index)

    def test_find_last_of_any(self):
        self.assertEqual(10, self.token_finder.find_last_of_any(["a", "b"]).index)

    def test_find_last_of_any_allow_space(self):
        self.assertEqual(10, self.token_finder.find_last_of_any(["a", "b"], allow_space_prefix=True).index)

    def test_find_last_of_any_in_scope(self):
        scope = Scope(0, 8, self.tokens)
        self.assertEqual(7, self.token_finder.find_last_of_any(["a", "b"], scope).index)

    def test_find_last_of_any_in_scope_allow_space(self):
        scope = Scope(0, 8, self.tokens)
        self.assertEqual(7, self.token_finder.find_last_of_any(["a", "b"], scope, allow_space_prefix=True).index)

    def test_find_all(self):
        self.assertEqual([6, 9], [token.index for token in self.token_finder.find_all("a")])

    def test_find_all_of_any(self):
        self.assertEqual([6, 7, 9, 10], [token.index for token in self.token_finder.find_all_of_any(["a", "b"])])