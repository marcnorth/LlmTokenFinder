import tempfile
import unittest
import torch
from transformer_lens import HookedTransformer
from llm_inspect import ActivationDataset, ActivationDatasetGenerator, ActivationGeneratorInput, ActivationProbe, AttentionHead


class AblationTest(unittest.TestCase):

    def test_generate_residual_stream_dataset(self):
        llm = HookedTransformer.from_pretrained("gpt2")
        def test_input_generator():
            yield ActivationGeneratorInput("first input text", 2, 3)
            yield ActivationGeneratorInput("second input text", 0, 4)
        class_labels = ["class_0", "class_1", "class_2", "class_3", "class_4"]
        residual_stream_dataset_generator = ActivationDatasetGenerator.create_residual_stream_generator(
            llm=llm,
            layer=1,
            input_generator=test_input_generator,
            class_labels=class_labels
        )
        with tempfile.TemporaryFile(mode="r+", encoding="utf-8") as file:
            residual_stream_dataset_generator.generate_and_save_to(file)
            loaded_dataset = ActivationDataset.load_from_file(file, device=llm.cfg.device)
        self.assertIsInstance(loaded_dataset, torch.utils.data.Dataset)
        self.assertEqual(2, len(loaded_dataset))
        self.assertIsInstance(loaded_dataset[0][0], torch.Tensor)
        self.assertIsInstance(loaded_dataset[0][1], torch.Tensor)
        self.assertEqual(3, loaded_dataset[0][1].item())
        self.assertEqual((llm.cfg.d_model,), loaded_dataset[0][0].shape)
        self.assertEqual(4, loaded_dataset[1][1].item())
        self.assertEqual((llm.cfg.d_model,), loaded_dataset[1][0].shape)
        self.assertEqual(class_labels, loaded_dataset.class_labels)

    def test_generate_attention_head_output_dataset(self):
        llm = HookedTransformer.from_pretrained("gpt2")
        def test_input_generator():
            yield ActivationGeneratorInput("first input text", 2, 3)
            yield ActivationGeneratorInput("second input text", 0, 4)
        class_labels = ["class_0", "class_1", "class_2", "class_3", "class_4"]
        residual_stream_dataset_generator = ActivationDatasetGenerator.create_attention_head_output_generator(
            llm=llm,
            head=AttentionHead(1, 4),
            input_generator=test_input_generator,
            class_labels=class_labels
        )
        with tempfile.TemporaryFile(mode="r+", encoding="utf-8") as file:
            residual_stream_dataset_generator.generate_and_save_to(file)
            loaded_dataset = ActivationDataset.load_from_file(file, device=llm.cfg.device)
        self.assertIsInstance(loaded_dataset, torch.utils.data.Dataset)
        self.assertEqual(2, len(loaded_dataset))
        self.assertIsInstance(loaded_dataset[0][0], torch.Tensor)
        self.assertIsInstance(loaded_dataset[0][1], torch.Tensor)
        self.assertEqual(3, loaded_dataset[0][1].item())
        self.assertEqual((llm.cfg.d_model,), loaded_dataset[0][0].shape)
        self.assertEqual(4, loaded_dataset[1][1].item())
        self.assertEqual((llm.cfg.d_model,), loaded_dataset[1][0].shape)
        self.assertEqual(class_labels, loaded_dataset.class_labels)

    def test_meta_data(self):
        llm = HookedTransformer.from_pretrained("gpt2")
        def test_input_generator():
            yield ActivationGeneratorInput("first input text", 2, 3)
        class_labels = ["class_0", "class_1", "class_2", "class_3", "class_4"]
        residual_stream_dataset_generator = ActivationDatasetGenerator.create_residual_stream_generator(
            llm=llm,
            layer=1,
            input_generator=test_input_generator,
            class_labels=class_labels,
            meta_data={"additional_info": "test"}
        )
        with tempfile.TemporaryFile(mode="r+", encoding="utf-8") as file:
            residual_stream_dataset_generator.generate_and_save_to(file)
            loaded_dataset = ActivationDataset.load_from_file(file, device=llm.cfg.device)
        self.assertIsInstance(loaded_dataset, torch.utils.data.Dataset)
        self.assertEqual(1, loaded_dataset.meta_data["layer"])
        self.assertEqual(None, loaded_dataset.meta_data["head"])
        self.assertEqual(class_labels, loaded_dataset.meta_data["class_labels"])
        self.assertEqual("gpt2", loaded_dataset.meta_data["llm"])
        self.assertEqual("test", loaded_dataset.meta_data["additional_info"])

    def test_generate_with_ablation(self):
        llm = HookedTransformer.from_pretrained("gpt2")
        def test_input_generator():
            yield ActivationGeneratorInput("first input text", 2, 3)
            yield ActivationGeneratorInput("second input text", 0, 4)
        class_labels = ["class_0", "class_1", "class_2", "class_3", "class_4"]
        residual_stream_dataset_generator = ActivationDatasetGenerator.create_attention_head_output_generator(
            llm=llm,
            head=AttentionHead(1, 4),
            input_generator=test_input_generator,
            class_labels=class_labels
        )
        with tempfile.TemporaryFile(mode="r+", encoding="utf-8") as file:
            residual_stream_dataset_generator.generate_and_save_to(file, heads_to_ablate=[AttentionHead(1, 4)])
            loaded_dataset = ActivationDataset.load_from_file(file, device=llm.cfg.device)
        self.assertIsInstance(loaded_dataset, torch.utils.data.Dataset)
        self.assertEqual(2, len(loaded_dataset))
        self.assertIsInstance(loaded_dataset[0][0], torch.Tensor)
        self.assertIsInstance(loaded_dataset[0][1], torch.Tensor)
        self.assertEqual(3, loaded_dataset[0][1].item())
        self.assertEqual((llm.cfg.d_model,), loaded_dataset[0][0].shape)
        self.assertEqual(4, loaded_dataset[1][1].item())
        self.assertEqual((llm.cfg.d_model,), loaded_dataset[1][0].shape)
        self.assertEqual(class_labels, loaded_dataset.class_labels)

    def test_training_probe(self):
        # Create dataset
        llm = HookedTransformer.from_pretrained("gpt2")
        def test_input_generator():
            for _ in range(50):
                yield ActivationGeneratorInput("the cat some words", 1, 0)
                yield ActivationGeneratorInput("a cat some words", 1, 0)
                yield ActivationGeneratorInput("the dog input text", 1, 1)
                yield ActivationGeneratorInput("a dog some words", 1, 1)
        class_labels = ["cat", "dog"]
        residual_stream_dataset_generator = ActivationDatasetGenerator.create_residual_stream_generator(
            llm=llm,
            layer=2,
            input_generator=test_input_generator,
            class_labels=class_labels
        )
        with tempfile.TemporaryFile(mode="r+", encoding="utf-8") as file:
            residual_stream_dataset_generator.generate_and_save_to(file)
            loaded_dataset = ActivationDataset.load_from_file(file, device=llm.cfg.device)
        # Train probe
        probe, *_ = loaded_dataset.train_probe(
            num_epochs=10,
            learning_rate=0.01,
            device=llm.cfg.device,
        )
        self.assertEqual(loaded_dataset.activation_dim, probe.linear1.in_features)
        self.assertEqual(len(class_labels), probe.linear2.out_features)
        # Test
        probe.eval()
        with torch.no_grad():
            # Cat input
            cat_input_ids = llm.tokenizer.encode("A cat some words", return_tensors="pt", add_special_tokens=True).to(llm.cfg.device)
            _, cat_cache = llm.run_with_cache(cat_input_ids, names_filter=lambda name: name == f"blocks.2.hook_resid_post", remove_batch_dim=True)
            cache_residual_stream = cat_cache["resid_post", 2][1].to(llm.cfg.device)
            cat_prediction_logits = probe(cache_residual_stream)
            cat_prediction = torch.argmax(cat_prediction_logits, dim=-1)
            self.assertEqual(0, cat_prediction.item())
            # Dog input
            dog_input_ids = llm.tokenizer.encode("A dog input text", return_tensors="pt", add_special_tokens=True).to(llm.cfg.device)
            _, dog_cache = llm.run_with_cache(dog_input_ids, names_filter=lambda name: name == f"blocks.2.hook_resid_post", remove_batch_dim=True)
            cache_residual_stream = dog_cache["resid_post", 2][1].to(llm.cfg.device)
            dog_prediction_logits = probe(cache_residual_stream)
            dog_prediction = torch.argmax(dog_prediction_logits, dim=-1)
            self.assertEqual(1, dog_prediction.item())

    def test_probe_save_load(self):
        # Create dataset
        llm = HookedTransformer.from_pretrained("gpt2")
        def test_input_generator():
            for _ in range(10):
                yield ActivationGeneratorInput("the cat some words", 1, 0)
                yield ActivationGeneratorInput("a cat some words", 1, 0)
                yield ActivationGeneratorInput("the dog input text", 1, 1)
                yield ActivationGeneratorInput("a dog some words", 1, 1)
        class_labels = ["cat", "dog"]
        residual_stream_dataset_generator = ActivationDatasetGenerator.create_residual_stream_generator(
            llm=llm,
            layer=2,
            input_generator=test_input_generator,
            class_labels=class_labels
        )
        with tempfile.TemporaryFile(mode="r+", encoding="utf-8") as file:
            residual_stream_dataset_generator.generate_and_save_to(file)
            loaded_dataset = ActivationDataset.load_from_file(file, device=llm.cfg.device)
        # Train probe
        probe, *_ = loaded_dataset.train_probe(
            num_epochs=1,
            learning_rate=0.01,
            device=llm.cfg.device,
        )
        with tempfile.TemporaryFile(mode="r+b") as file:
            probe.save_to_file(file)
            loaded_probe = ActivationProbe.load_from_file(file, device=llm.cfg.device)
        self.assertEqual(probe.linear1.in_features, loaded_probe.linear1.in_features)
        self.assertEqual(probe.linear2.out_features, loaded_probe.linear2.out_features)
        self.assertEqual(probe.linear1.out_features, loaded_probe.linear1.out_features)
        self.assertEqual(probe._activation_dataset_meta_data, loaded_probe._activation_dataset_meta_data)
        self.assertEqual(probe._meta_data, loaded_probe._meta_data)
        for param1, param2 in zip(probe.parameters(), loaded_probe.parameters()):
            self.assertTrue(torch.equal(param1, param2), "Probe parameters do not match after save/load.")
