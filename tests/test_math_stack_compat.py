import importlib
import unittest
from types import SimpleNamespace
from pathlib import Path
from unittest import mock

from hydra import compose, initialize_config_dir
from hydra.core.global_hydra import GlobalHydra
from transformers.tokenization_utils_base import BatchEncoding


class MathStackCompatTest(unittest.TestCase):
    def test_math_path_imports_without_tapeagents(self):
        modules = [
            "pipelinerl.llm",
            "pipelinerl.actor",
            "pipelinerl.async_llm",
            "pipelinerl.preprocess",
            "pipelinerl.launch",
            "pipelinerl.vllm1",
            "pipelinerl.finetune.checkpoints",
        ]
        for module_name in modules:
            with self.subTest(module=module_name):
                importlib.import_module(module_name)

    def test_math_config_composes_with_v1_enabled(self):
        conf_dir = Path(__file__).resolve().parents[1] / "conf"
        GlobalHydra.instance().clear()
        with initialize_config_dir(config_dir=str(conf_dir), version_base=None):
            cfg = compose(config_name="math")
        self.assertTrue(cfg.vllm_config.use_v1)

    def test_v1_launcher_filters_legacy_vllm_flags(self):
        launch = importlib.import_module("pipelinerl.launch")
        cmd = ["python", "-m", "pipelinerl.entrypoints.run_vllm1"]
        launch._append_vllm_kwargs(
            cmd,
            use_v1=True,
            kwargs={
                "tensor-parallel-size": 2,
                "disable-log-requests": "",
                "num-scheduler-steps": 1,
            },
        )
        self.assertIn("--tensor-parallel-size", cmd)
        self.assertIn("2", cmd)
        self.assertNotIn("--disable-log-requests", cmd)
        self.assertNotIn("--num-scheduler-steps", cmd)

    def test_make_training_text_accepts_batch_encoding_prompt_ids(self):
        async_llm = importlib.import_module("pipelinerl.async_llm")
        llm_module = importlib.import_module("pipelinerl.llm")

        class StubTokenizer:
            bos_token = None
            eos_token = "</s>"

            def apply_chat_template(self, conversation, tokenize=True, **kwargs):
                if tokenize is False:
                    return "|".join(f"{message['role']}:{message['content']}" for message in conversation)
                length = len(conversation) + 2
                return BatchEncoding({"input_ids": list(range(100, 100 + length))})

        llm = llm_module.TrainableLLM(
            base_url="http://localhost:8000",
            model_name="stub-model",
            collect_logprobs=True,
        )
        llm.tokenizer = StubTokenizer()
        llm_call = llm_module.LLMCall(
            prompt=llm_module.Prompt(messages=[{"role": "user", "content": "Solve x+1=2"}]),
            output=llm_module.LLMOutput(content="x=1"),
            prompt_length_tokens=3,
            output_length_tokens=2,
            logprobs=[
                llm_module.TokenLogprob(token_id=201, logprob=-0.1),
                llm_module.TokenLogprob(token_id=202, logprob=-0.2),
            ],
        )

        training_text = async_llm.make_training_text(llm, llm_call)
        self.assertEqual(training_text.input_ids, [100, 101, 102, 201, 202])
        self.assertEqual(training_text.labels[-2:], [201, 202])

    def test_qwen35_weight_name_translation_handles_prefixes_and_tied_weights(self):
        vllm1 = importlib.import_module("pipelinerl.vllm1")

        self.assertEqual(
            vllm1._translate_weight_name_for_vllm("Qwen3_5ForConditionalGeneration", "model.embed_tokens.weight"),
            ("language_model.model.embed_tokens.weight", None),
        )
        self.assertEqual(
            vllm1._translate_weight_name_for_vllm(
                "Qwen3_5ForConditionalGeneration",
                "model.layers.0.linear_attn.in_proj_qkv.weight",
            ),
            ("language_model.model.layers.0.linear_attn.in_proj_qkv.weight", None),
        )
        self.assertEqual(
            vllm1._translate_weight_name_for_vllm(
                "Qwen3_5ForConditionalGeneration",
                "model.layers.0.linear_attn.in_proj_z.weight",
            ),
            ("language_model.model.layers.0.linear_attn.in_proj_z.weight", None),
        )
        self.assertEqual(
            vllm1._translate_weight_name_for_vllm(
                "Qwen3_5ForConditionalGeneration",
                "model.layers.0.mlp.gate_proj.weight",
            ),
            ("language_model.model.layers.0.mlp.gate_proj.weight", None),
        )
        self.assertEqual(
            vllm1._translate_weight_name_for_vllm("Qwen3_5ForConditionalGeneration", "lm_head.weight"),
            (None, None),
        )

    def test_qwen35_runtime_guard_requires_ninja_on_sm90(self):
        vllm1 = importlib.import_module("pipelinerl.vllm1")
        config = SimpleNamespace(
            model_config=SimpleNamespace(
                hf_config=SimpleNamespace(model_type="qwen3_5"),
            )
        )

        with (
            mock.patch.object(vllm1.current_platform, "is_device_capability", return_value=True),
            mock.patch.object(vllm1.shutil, "which", return_value=None),
        ):
            with self.assertRaisesRegex(RuntimeError, "requires the `ninja` executable"):
                vllm1._ensure_qwen35_runtime_dependencies(config)

    def test_qwen35_runtime_guard_allows_ninja_on_sm90(self):
        vllm1 = importlib.import_module("pipelinerl.vllm1")
        config = SimpleNamespace(
            model_config=SimpleNamespace(
                hf_config=SimpleNamespace(model_type="qwen3_5"),
            )
        )

        with (
            mock.patch.object(vllm1.current_platform, "is_device_capability", return_value=True),
            mock.patch.object(vllm1.shutil, "which", return_value="/usr/bin/ninja"),
        ):
            vllm1._ensure_qwen35_runtime_dependencies(config)

    def test_qwen35_runtime_guard_skips_non_qwen35_models(self):
        vllm1 = importlib.import_module("pipelinerl.vllm1")
        config = SimpleNamespace(
            model_config=SimpleNamespace(
                hf_config=SimpleNamespace(model_type="qwen2"),
            )
        )

        with (
            mock.patch.object(vllm1.current_platform, "is_device_capability", return_value=True),
            mock.patch.object(vllm1.shutil, "which", return_value=None),
        ):
            vllm1._ensure_qwen35_runtime_dependencies(config)

    def test_remote_model_source_is_prefetched_once_then_resolved_locally(self):
        checkpoints = importlib.import_module("pipelinerl.finetune.checkpoints")

        class StubAccelerator:
            is_main_process = True

            def wait_for_everyone(self):
                return None

        calls: list[dict] = []

        def fake_snapshot_download(repo_id, revision=None, local_files_only=False):
            calls.append(
                {
                    "repo_id": repo_id,
                    "revision": revision,
                    "local_files_only": local_files_only,
                }
            )
            return "/tmp/qwen35-snapshot"

        with (
            mock.patch.object(checkpoints, "get_accelerator", return_value=StubAccelerator()),
            mock.patch.object(checkpoints, "snapshot_download", side_effect=fake_snapshot_download),
        ):
            resolved = checkpoints._resolve_model_source("Qwen/Qwen3.5-4B", "main")

        self.assertEqual(resolved, "/tmp/qwen35-snapshot")
        self.assertEqual(
            calls,
            [
                {
                    "repo_id": "Qwen/Qwen3.5-4B",
                    "revision": "main",
                    "local_files_only": False,
                },
                {
                    "repo_id": "Qwen/Qwen3.5-4B",
                    "revision": "main",
                    "local_files_only": True,
                },
            ],
        )

    def test_gsm8k_qwen35_override_caps_vllm_context(self):
        conf_dir = Path(__file__).resolve().parents[1] / "conf"
        GlobalHydra.instance().clear()
        with initialize_config_dir(config_dir=str(conf_dir), version_base=None):
            cfg = compose(
                config_name="gsm8k",
                overrides=["model_path=Qwen/Qwen3.5-2B"],
            )
        self.assertEqual(cfg.vllm_config.vllm_kwargs["max-model-len"], cfg.finetune.seq_length)


if __name__ == "__main__":
    unittest.main()
