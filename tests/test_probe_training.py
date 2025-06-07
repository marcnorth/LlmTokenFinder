import tempfile
import unittest
import torch
from transformer_lens import HookedTransformer

from activation_probing.activation_dataset import ActivationDataset
from activation_probing.activation_dataset_generator import ActivationDatasetGenerator, ActivationGeneratorInput
from llm_token_finder.activation_analyser import AttentionHead


class AblationTest(unittest.TestCase):

    def test_generate_residual_stream_dataset(self):
        llm = HookedTransformer.from_pretrained("gpt2-small")
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

    def test_meta_data(self):
        residual_stream_dataset_generator = ActivationDatasetGenerator(
            file_handler=file,
            meta_data={"name": "test_name", "experiment": "test_experiment"},
            llm=self.llm,
            layer=1,
            input_generator=test_input_generator, # Iterator that returns the input text for each forward pass (and terminates when the iterator is exhausted). Also the token position for each forward pass.
        )
        self.assertEqual(ActivationType.RESIDUAL_STREAM, residual_stream_dataset_generator.meta_data["activation_type"])
        self.assertEqual(1, residual_stream_dataset_generator.meta_data["layer_index"])
        self.assertEqual(None, residual_stream_dataset_generator.meta_data["head_index"])
        self.assertEqual("test_name", residual_stream_dataset_generator.meta_data["name"])
        self.assertEqual(self.llm.cfg.name, residual_stream_dataset_generator.meta_data["llm"])

    def test_generate_attention_head_output_dataset(self):
        attention_head_output_dataset = ActivationDatasetGenerator.create_attention_head_output_generator(
            llm=self.llm,
            head=AttentionHead(2, 4),
            input_generator=None
        )