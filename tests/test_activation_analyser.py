import unittest
import torch
from llm_token_finder import TokenFinder, Scope, ActivationAnalyzer
from transformer_lens import utils, HookedTransformer, ActivationCache


class ActivationAnalyserTest(unittest.TestCase):

    def setUp(self):
        self.llm = HookedTransformer.from_pretrained("gpt2-small")
        self.text = "The quick brown fox jumps over the lazy dog."
        self.token_finder = TokenFinder.create_from_tokenizer(self.text, self.llm.tokenizer)
        self.activation_analyser = ActivationAnalyzer.from_forward_pass(self.llm, self.text)

    def test_find_heads_matching_criteria(self):
        # Every token looks at the previous token
        looks_at_previous_token = lambda attention: torch.all(attention.argmax(-1)[1:] == torch.arange(attention.shape[0]-1))
        matching_heads = self.activation_analyser.find_heads_matching_criteria(looks_at_previous_token)
        # Should only match 4, 11
        self.assertEqual(len(matching_heads), 1)
        self.assertEqual(matching_heads[0].layer, 4)
        self.assertEqual(matching_heads[0].head, 11)

    def test_heads_where_query_looks_at_value(self):
        # Find heads where the word "fox" looks at the word "quick" more than any other token
        token_finder = TokenFinder.create_from_tokenizer(self.text, self.llm.tokenizer)
        quick = token_finder.find_first("quick", allow_space_prefix=True)
        fox = token_finder.find_first("fox", allow_space_prefix=True)
        matching_heads = self.activation_analyser.find_heads_where_query_looks_at_value(fox, quick)
        self.assertEqual(len(matching_heads), 3)
        self.assertEqual(matching_heads[0].layer, 2)
        self.assertEqual(matching_heads[0].head, 8)

    def test_heads_where_query_looks_at_values(self):
        # Find heads where the word "fox" looks at the words "quick" and "brown" more than any other token
        token_finder = TokenFinder.create_from_tokenizer(self.text, self.llm.tokenizer)
        quick = token_finder.find_first("quick", allow_space_prefix=True)
        brown = token_finder.find_first("brown", allow_space_prefix=True)
        fox = token_finder.find_first("fox", allow_space_prefix=True)
        matching_heads = self.activation_analyser.find_heads_where_query_looks_at_values(fox, (quick, brown))
        self.assertEqual(len(matching_heads), 2)
        self.assertEqual(matching_heads[0].layer, 2)
        self.assertEqual(matching_heads[0].head, 9)
