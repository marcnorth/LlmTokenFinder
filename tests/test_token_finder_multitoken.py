import unittest
from llm_inspect import TokenFinder, TokenRange, Token


class TokenFinderTest(unittest.TestCase):

    def setUp(self):
        self.tokens = [
            "Ġa", # 0
            "Ġb", # 1
            "Ġc", # 2
            "Ġa", # 3
            "Ġb", # 4
            "Ġc", # 5
            "ab", # 6
            "c", # 7
            "ab", # 8
            "c" # 9
        ]
        self.token_finder = TokenFinder(self.tokens, space_token="Ġ")

    def test_find_first_multitoken(self):
        result = self.token_finder.find_first_range("abc", allow_space_prefix=False)
        self.assertEqual(result.start, 6)
        self.assertEqual(result.end, 7)
        self.assertEqual(result[0].index, 6)
        self.assertEqual(result[0].value, "ab")
        self.assertEqual(result[1].index, 7)
        self.assertEqual(result[1].value, "c")

    def test_wont_match_partial_tokens_by_default(self):
        full_match = self.token_finder.find_first_range("cab", allow_space_prefix=False)
        self.assertEqual(full_match.start, 7)
        self.assertEqual(full_match.end, 8)
        with self.assertRaises(ValueError):
            self.token_finder.find_first_range("ca", allow_space_prefix=False)

    def test_with_space_prefix(self):
        without_prefix = self.token_finder.find_first_range("cab", allow_space_prefix=False)
        self.assertEqual(without_prefix.start, 7)
        self.assertEqual(without_prefix.end, 8)
        with_prefix = self.token_finder.find_first_range("cab", allow_space_prefix=True)
        self.assertEqual(with_prefix.start, 5)
        self.assertEqual(with_prefix.end, 6)

    def test_start_with_space(self):
        without_prefix = self.token_finder.find_first_range(" a b c", allow_space_prefix=True)
        self.assertEqual(without_prefix.start, 0)
        self.assertEqual(without_prefix.end, 2)

    def test_with_spaces_in_between(self):
        result = self.token_finder.find_first_range("a b c", allow_space_prefix=True)
        self.assertEqual(result.start, 0)
        self.assertEqual(result.end, 2)

    def test_returns_token_if_range_is_single_token(self):
        result = self.token_finder.find_first_range("ab", allow_space_prefix=False)
        self.assertEqual(result.start, 6)
        self.assertEqual(result.end, 6)
        self.assertIsInstance(result, TokenRange)
        self.assertIsInstance(result, Token)
        self.assertEqual(result.index, 6)
        self.assertEqual(result.value, "ab")

    def test_allow_partial_match(self):
        result = self.token_finder.find_first_range("abca", allow_space_prefix=False, allow_partial_final_token_match=True)
        self.assertEqual(result.start, 6)
        self.assertEqual(result.end, 8)
        self.assertEqual("abcab", result.to_string())
