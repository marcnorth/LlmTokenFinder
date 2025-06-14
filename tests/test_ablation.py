import unittest
import torch
from transformer_lens import HookedTransformer
from llm_inspect import AttentionHeadFinder, TokenFinder, AblationLlm, AttentionHead


class AblationTest(unittest.TestCase):

    def setUp(self):
        self.llm = HookedTransformer.from_pretrained("gpt2-small")
        self.text = "The quick brown fox jumps over the lazy dog."
        self.token_finder = TokenFinder.create_from_tokenizer(self.text, self.llm.tokenizer)
        self.activation_analyser = AttentionHeadFinder.from_forward_pass(self.llm, self.text)

    def test_attention_head_ablation(self):
        ablation_llm = AblationLlm(self.llm)
        ablation_llm.llm.cfg.use_attn_result = True
        heads_to_ablate = [
            AttentionHead(1, 3),
            AttentionHead(1, 6)
        ]
        unablated_logits, unablated_cache = ablation_llm.forward(self.text)
        ablated_logits, ablated_cache = ablation_llm.forward(self.text, heads_to_ablate=heads_to_ablate)
        self.assertFalse(torch.equal(unablated_logits, ablated_logits))

        unablated_head_output_1 = unablated_cache["result", 1][0][-1][3]
        unablated_head_output_2 = unablated_cache["result", 1][0][-1][6]

        ablated_head_output_1 = ablated_cache["result", 1][0][-1][3]
        ablated_head_output_2 = ablated_cache["result", 1][0][-1][6]

        self.assertFalse(torch.all(unablated_head_output_1 == 0))
        self.assertFalse(torch.all(unablated_head_output_2 == 0))
        self.assertTrue(torch.all(ablated_head_output_1 == 0))
        self.assertTrue(torch.all(ablated_head_output_2 == 0))

    def test_ablation_between_tokens(self):
        ablation_llm = AblationLlm(self.llm)

        # Ablate all movement between the tokens "quick" and "fox"
        quick_token = self.token_finder.find_first("quick", allow_space_prefix=True)
        fox_token = self.token_finder.find_first("fox", allow_space_prefix=True)

        unablated_logits, unablated_cache = ablation_llm.forward(self.text)
        ablated_logits, ablated_cache = ablation_llm.forward(self.text, token_movement_to_ablate=[(quick_token, fox_token)])

        self.assertFalse(torch.equal(unablated_logits, ablated_logits))

        for layer_index in range(self.llm.cfg.n_layers):
            for head_index in range(self.llm.cfg.n_heads):
                unablated_attention_pattern = unablated_cache["pattern", layer_index][0, head_index]
                ablated_attention_pattern = ablated_cache["pattern", layer_index][0, head_index]

                # Check that the attention pattern for the head is not all zeros between 'quick' and 'fox' without ablation
                self.assertFalse(torch.all(unablated_attention_pattern == 0))
                self.assertFalse(torch.all(unablated_attention_pattern[fox_token.index][quick_token.index] == 0))

                # Check that the attention pattern for the head is all zeros between 'quick' and 'fox' with ablation
                self.assertFalse(torch.all(ablated_attention_pattern == 0))
                self.assertTrue(torch.all(ablated_attention_pattern[fox_token.index][quick_token.index] == 0))

    def test_ablation_logits(self):
        # If we ablate the movement from a token to every subsequent token, that token should not affect the logits of subsequent tokens.
        ablation_llm = AblationLlm(self.llm)
        # WET
        word_to_ablate = "wet"
        text = f"I have a {word_to_ablate} shirt. My shirt is"
        token_finder = TokenFinder.create_from_tokenizer(text, self.llm.tokenizer)
        tokens = self.llm.tokenizer.tokenize(text, add_special_tokens=True)
        from_token = token_finder.find_first(word_to_ablate, allow_space_prefix=True)
        # Get unablated logits with "wet"
        wet_unablated_logits, _ = ablation_llm.forward(text)
        # Get ablated logits with "wet"
        to_ablate = [(from_token.index, i) for i in range(from_token.index+1, len(tokens))]  # Ablate movement from token index 4 to indices 5-9
        wet_ablated_logits, _ = ablation_llm.forward(text, token_movement_to_ablate=to_ablate)
        self.assertFalse(torch.equal(wet_unablated_logits, wet_ablated_logits))
        # TIGHT
        word_to_ablate = "tight"
        text = f"I have a {word_to_ablate} shirt. My shirt is"
        token_finder = TokenFinder.create_from_tokenizer(text, self.llm.tokenizer)
        tokens = self.llm.tokenizer.tokenize(text, add_special_tokens=True)
        from_token = token_finder.find_first(word_to_ablate, allow_space_prefix=True)
        # Get unablated logits with "tight"
        tight_unablated_logits, _ = ablation_llm.forward(text)
        # Get ablated logits with "tight"
        to_ablate = [(from_token.index, i) for i in range(from_token.index+1, len(tokens))]  # Ablate movement from token index 4 to indices 5-9
        tight_ablated_logits, _ = ablation_llm.forward(text, token_movement_to_ablate=to_ablate)
        self.assertFalse(torch.equal(tight_unablated_logits, tight_ablated_logits))
        # Ablated logits should not be affected "wet" or "tight", so they should be the same everywhere except for the token being ablated.
        self.assertFalse(torch.all(wet_ablated_logits[:, from_token.index] == tight_ablated_logits[:, from_token.index]))
        self.assertTrue(torch.all(wet_ablated_logits[:, :from_token.index] == tight_ablated_logits[:, :from_token.index]))
        self.assertTrue(torch.all(wet_ablated_logits[:, from_token.index + 1:] == tight_ablated_logits[:, from_token.index + 1:]))
