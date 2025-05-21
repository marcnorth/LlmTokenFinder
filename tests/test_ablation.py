import unittest
import torch
from transformer_lens import HookedTransformer
from llm_token_finder import ActivationAnalyzer, TokenFinder
from llm_token_finder.ablation import AblationLlm
from llm_token_finder.activation_analyser import AttentionHead


class AblationTest(unittest.TestCase):

    def setUp(self):
        self.llm = HookedTransformer.from_pretrained("gpt2-small")
        self.text = "The quick brown fox jumps over the lazy dog."
        self.token_finder = TokenFinder.create_from_tokenizer(self.text, self.llm.tokenizer)
        self.activation_analyser = ActivationAnalyzer.from_forward_pass(self.llm, self.text)

    def test_basic_ablation(self):
        ablation_llm = AblationLlm(self.llm)
        ablation_llm.llm.cfg.use_attn_result = True
        heads_to_ablate = [
            AttentionHead(1, 3),
            AttentionHead(1, 6)
        ]
        unablated_logits, unablated_cache = ablation_llm.forward(self.text)
        ablated_logits, ablated_cache = ablation_llm.forward(self.text, heads_to_ablate=heads_to_ablate)
        self.assertFalse(torch.equal(unablated_logits, ablated_logits))

        unablated_head_result_1 = unablated_cache["result", 1][0][-1][3]
        unablated_head_result_2 = unablated_cache["result", 1][0][-1][6]

        ablated_head_result_1 = ablated_cache["result", 1][0][-1][3]
        ablated_head_result_2 = ablated_cache["result", 1][0][-1][6]

        self.assertFalse(torch.all(unablated_head_result_1 == 0))
        self.assertFalse(torch.all(unablated_head_result_2 == 0))
        self.assertTrue(torch.all(ablated_head_result_1 == 0))
        self.assertTrue(torch.all(ablated_head_result_2 == 0))
