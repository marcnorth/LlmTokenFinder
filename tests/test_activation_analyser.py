import unittest
import torch

from transformers import AutoModel

from llm_token_finder import TokenFinder, Scope, ActivationAnalyzer
from transformer_lens import utils, HookedTransformer, ActivationCache


class ActivationAnalyserTest(unittest.TestCase):
    def test_find_heads_matching_criteria(self):
        model = HookedTransformer.from_pretrained("gpt2-small")
        text = "The quick brown fox jumps over the lazy dog."
        tokens = model.tokenizer.tokenize(text, add_special_tokens=True)
        token_ids = model.tokenizer.encode(text, return_tensors="pt")
        _, activation_cache = model.run_with_cache(token_ids)
        # Every token looks at the previous token
        looks_at_previous_token = lambda attention: torch.all(attention.argmax(-1)[1:] == torch.arange(attention.shape[0]-1))
        activation_analyser = ActivationAnalyzer(tokens, activation_cache)
        matching_heads = activation_analyser.find_heads_matching_criteria(looks_at_previous_token)
        # Should only match 4, 11
        self.assertEqual(len(matching_heads), 1)
        self.assertEqual(matching_heads[0].layer, 4)
        self.assertEqual(matching_heads[0].head, 11)