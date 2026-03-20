import importlib
import asyncio
import unittest
from pathlib import Path
from unittest import mock

from hydra import compose, initialize_config_dir
from hydra.core.global_hydra import GlobalHydra
from omegaconf import OmegaConf


class RCStackCompatTest(unittest.TestCase):
    @staticmethod
    def _conf_dir() -> Path:
        return Path(__file__).resolve().parents[1] / "conf"

    def _compose(self, config_name: str, overrides: list[str] | None = None):
        GlobalHydra.instance().clear()
        with initialize_config_dir(config_dir=str(self._conf_dir()), version_base=None):
            return compose(config_name=config_name, overrides=overrides or [])

    def test_rc_modules_import_without_tapeagents(self):
        modules = [
            "pipelinerl.rc_actor",
            "pipelinerl.entrypoints.run_rc_actor",
            "pipelinerl.test_rc_actor",
        ]
        for module_name in modules:
            with self.subTest(module=module_name):
                importlib.import_module(module_name)

    def test_rc_entrypoints_default_to_rc_smoke(self):
        run_rc_actor = importlib.import_module("pipelinerl.entrypoints.run_rc_actor")
        test_rc_actor = importlib.import_module("pipelinerl.test_rc_actor")

        self.assertEqual(run_rc_actor.DEFAULT_RC_CONFIG_NAME, "rc_smoke")
        self.assertEqual(test_rc_actor.DEFAULT_RC_TEST_CONFIG, "rc_smoke")

    def test_rc_configs_compose(self):
        for config_name in [
            "rc_smoke",
            "rc_tiny",
            "rc_tiny_flash_probe",
            "rc_test",
            "rc_proof_qwen3-4b-thinking_v18.00",
        ]:
            with self.subTest(config=config_name):
                self._compose(config_name)

    def test_rc_smoke_uses_qwen35_fast_models(self):
        launch = importlib.import_module("pipelinerl.launch")
        cfg = self._compose("rc_smoke", overrides=["output_dir=/tmp/rc-smoke"])

        self.assertEqual(cfg.model_path, "Qwen/Qwen3.5-0.8B")
        self.assertEqual(cfg.summarization_model_path, "Qwen/Qwen3.5-0.8B")
        self.assertTrue(cfg.actor.use_rc_stream)
        self.assertEqual(
            cfg.actor.rollout_policy,
            "pipelinerl.domains.math.rollouts.generate_math_rollout",
        )

        launch._apply_model_compat_overrides(cfg)

        self.assertIn("language-model-only", cfg.rc_actor_vllm_config.vllm_kwargs)
        self.assertIn("language-model-only", cfg.summarization_vllm_config.vllm_kwargs)

    def test_test_harness_prepares_world_map_from_world(self):
        test_rc_actor = importlib.import_module("pipelinerl.test_rc_actor")
        cfg = self._compose("rc_smoke", overrides=["output_dir=/tmp/rc-smoke"])

        with mock.patch("torch.cuda.device_count", return_value=8):
            cfg, world_map = test_rc_actor.prepare_config_for_test(cfg, Path("/tmp/rc-smoke"))

        self.assertEqual(cfg.me.llm_urls, "+".join(world_map.get_rc_actor_urls()))
        self.assertEqual(cfg.me.summarization_llm_urls, "+".join(world_map.get_summarization_urls()))
        self.assertTrue(world_map.get_rc_actor_urls())
        self.assertTrue(world_map.get_summarization_urls())

    def test_rc_smoke_uses_v1_entrypoint(self):
        test_rc_actor = importlib.import_module("pipelinerl.test_rc_actor")
        cfg = self._compose("rc_smoke", overrides=["output_dir=/tmp/rc-smoke"])
        self.assertEqual(
            test_rc_actor._get_vllm_entrypoint(cfg.rc_actor_vllm_config),
            "pipelinerl.entrypoints.run_vllm1",
        )

    def test_reduced_proof_v18_world_map_fits_eight_gpu_smoke(self):
        world_module = importlib.import_module("pipelinerl.world")
        cfg = self._compose(
            "rc_proof_qwen3-4b-thinking_v18.00",
            overrides=[
                "output_dir=/tmp/rc-proof-v18",
                "world.rc_actor_fraction=4",
                "world.summarization_fraction=4",
                "world.actor_fraction=0",
                "world.finetune_fraction=0",
                "world.env_replicas=1",
                "rc_actor.llm_max_rollouts=2",
                "rc_actor.summarization_max_rollouts=2",
                "eval_only=true",
            ],
        )

        with mock.patch("torch.cuda.device_count", return_value=8):
            world_map = world_module.WorldMap(cfg, verbose=False)

        rc_actor_jobs = [job for job in world_map.get_all_jobs() if job.kind == "rc_actor_llm"]
        summarization_jobs = [job for job in world_map.get_all_jobs() if job.kind == "summarization_llm"]
        actor_jobs = [job for job in world_map.get_all_jobs() if job.kind == "actor_llm"]
        finetune_jobs = [job for job in world_map.get_all_jobs() if job.kind == "finetune"]

        self.assertEqual(len(rc_actor_jobs), 4)
        self.assertEqual(len(summarization_jobs), 4)
        self.assertFalse(actor_jobs)
        self.assertFalse(finetune_jobs)
        self.assertEqual([job.port for job in rc_actor_jobs], [8000, 8001, 8002, 8003])
        self.assertEqual([job.port for job in summarization_jobs], [8204, 8205, 8206, 8207])

    def test_rc_tiny_world_map_matches_single_node_smoke_layout(self):
        launch = importlib.import_module("pipelinerl.launch")
        world_module = importlib.import_module("pipelinerl.world")
        cfg = self._compose("rc_tiny", overrides=["output_dir=/tmp/rc-tiny"])

        launch._apply_model_compat_overrides(cfg)

        with mock.patch("torch.cuda.device_count", return_value=8):
            world_map = world_module.WorldMap(cfg, verbose=False)

        rc_actor_jobs = [job for job in world_map.get_all_jobs() if job.kind == "rc_actor_llm"]
        summarization_jobs = [job for job in world_map.get_all_jobs() if job.kind == "summarization_llm"]
        actor_jobs = [job for job in world_map.get_all_jobs() if job.kind == "actor_llm"]
        finetune_jobs = [job for job in world_map.get_all_jobs() if job.kind == "finetune"]

        self.assertEqual(len(rc_actor_jobs), 2)
        self.assertEqual(len(summarization_jobs), 1)
        self.assertEqual(len(actor_jobs), 3)
        self.assertEqual(len(finetune_jobs), 1)
        self.assertEqual(len(finetune_jobs[0].gpus), 2)
        self.assertEqual(cfg.finetune.seq_parallel, 1)
        self.assertEqual(cfg.finetune.gradient_accumulation_passes, 2)
        self.assertEqual(cfg.finetune.rl.temperature, 1.0)
        self.assertTrue(cfg.finetune.allow_qwen35_text_flash_probe)
        self.assertTrue(cfg.finetune.use_flash_attention)
        self.assertFalse(cfg.finetune.seq_packing)
        self.assertEqual(cfg.finetune.attn_implementation, "flash_attention_2")
        self.assertEqual(cfg.vllm_config.vllm_kwargs["attention-backend"], "FLASH_ATTN")
        self.assertEqual(cfg.rc_actor_vllm_config.vllm_kwargs["attention-backend"], "FLASH_ATTN")
        self.assertEqual(cfg.summarization_vllm_config.vllm_kwargs["attention-backend"], "FLASH_ATTN")
        self.assertIn("language-model-only", cfg.vllm_config.vllm_kwargs)
        self.assertIn("language-model-only", cfg.rc_actor_vllm_config.vllm_kwargs)
        self.assertIn("language-model-only", cfg.summarization_vllm_config.vllm_kwargs)
        self.assertEqual(cfg.llm_grader.vllm_kwargs["max-num-batched-tokens"], 4096)
        self.assertEqual(cfg.llm_grader.vllm_kwargs["max-num-seqs"], 4)
        self.assertEqual(cfg.llm_grader.vllm_kwargs["max-model-len"], 32768)
        self.assertEqual(cfg.llm_grader.vllm_kwargs["gpu-memory-utilization"], 0.85)
        self.assertEqual(cfg.llm_grader.sampling_kwargs.max_output_tokens, 2048)
        self.assertEqual(cfg.llm_grader.reasoning_delimiters, [])
        self.assertEqual(cfg.train_dataset_names[0].hub_id, "lm-provers/olympiads-proof-schema")
        self.assertEqual(cfg.train_dataset_names[0].split, "train")
        self.assertEqual(list(cfg.test_dataset_names), ["imo_proof_bench"])

    def test_rc_tiny_flash_probe_preserves_flash_attention_and_packing(self):
        launch = importlib.import_module("pipelinerl.launch")
        cfg = self._compose("rc_tiny_flash_probe", overrides=["output_dir=/tmp/rc-tiny-flash-probe"])

        launch._apply_model_compat_overrides(cfg)

        self.assertTrue(cfg.finetune.allow_qwen35_text_flash_probe)
        self.assertTrue(cfg.finetune.use_flash_attention)
        self.assertTrue(cfg.finetune.seq_packing)
        self.assertEqual(cfg.finetune.attn_implementation, "flash_attention_2")
        self.assertEqual(cfg.vllm_config.vllm_kwargs["attention-backend"], "FLASH_ATTN")
        self.assertEqual(cfg.rc_actor_vllm_config.vllm_kwargs["attention-backend"], "FLASH_ATTN")
        self.assertEqual(cfg.summarization_vllm_config.vllm_kwargs["attention-backend"], "FLASH_ATTN")
        self.assertIn("language-model-only", cfg.vllm_config.vllm_kwargs)
        self.assertIn("language-model-only", cfg.rc_actor_vllm_config.vllm_kwargs)
        self.assertIn("language-model-only", cfg.summarization_vllm_config.vllm_kwargs)

    def test_finetune_qwen35_probe_flag_controls_override(self):
        finetune_loop = importlib.import_module("pipelinerl.finetune_loop")

        base_args = {
            "model_class": "causal-language-modeling",
            "config_name": "Qwen/Qwen3.5-2B",
            "use_flash_attention": True,
            "attn_implementation": "flash_attention_2",
            "seq_packing": True,
        }
        default_args = OmegaConf.create(base_args)
        probe_args = OmegaConf.create({**base_args, "allow_qwen35_text_flash_probe": True})

        finetune_loop._apply_finetune_model_compat_overrides(default_args)
        finetune_loop._apply_finetune_model_compat_overrides(probe_args)

        self.assertFalse(default_args.use_flash_attention)
        self.assertEqual(default_args.attn_implementation, "sdpa")
        self.assertFalse(default_args.seq_packing)
        self.assertTrue(probe_args.use_flash_attention)
        self.assertEqual(probe_args.attn_implementation, "flash_attention_2")
        self.assertTrue(probe_args.seq_packing)

    def test_rc_rollout_ignores_null_schema_for_gsm8k(self):
        llm_module = importlib.import_module("pipelinerl.llm")
        rollout_models = importlib.import_module("pipelinerl.rollouts")
        rollouts = importlib.import_module("pipelinerl.domains.math.rollouts")

        cfg = OmegaConf.create(
            {
                "rc_actor": {
                    "system_prompt": None,
                    "task_template": "{task}",
                    "discount_factor": 1.0,
                },
                "actor": {},
                "llm_grader": {},
                "llm": {"parameters": {"max_tokens": 16}},
                "rewards": {
                    "wrong_answer_not_finished": 0.0,
                    "wrong_answer_finished": 0.0,
                    "no_answer_not_finished": 0.0,
                    "no_answer_finished": 0.0,
                    "unparsable_not_finished": 0.0,
                    "unparsable_finished": 0.0,
                    "correct_answer_not_finished": 1.0,
                    "correct_answer_finished": 1.0,
                },
                "jobs": [
                    {
                        "kind": "environment",
                        "idx": 0,
                        "replica_idx": 0,
                        "node_rank": 0,
                        "hostname": "localhost",
                        "port": 7777,
                        "gpus": [],
                        "url": "",
                    }
                ],
                "wandb": {"use_wandb": False},
            }
        )
        llm = llm_module.TrainableLLM(
            base_url="http://localhost:8000",
            model_name="stub-model",
            parameters={"max_tokens": 16},
        )
        llm_call = llm_module.LLMCall(
            prompt=llm_module.Prompt(messages=[{"role": "user", "content": "2+2?"}]),
            output=llm_module.LLMOutput(content="4"),
            prompt_length_tokens=1,
            output_length_tokens=1,
        )
        training_text = rollout_models.TrainingText(
            text="Q4",
            output_text="4",
            n_predicted=1,
            finished=True,
            prompt_tokens=1,
            output_tokens=1,
        )

        async def invoke():
            with (
                mock.patch.object(rollouts, "llm_async_generate", new=mock.AsyncMock(return_value=llm_call)),
                mock.patch.object(rollouts, "make_training_text", return_value=training_text),
                mock.patch.object(rollouts, "verify_answer_rpc", new=mock.AsyncMock(return_value="correct")),
                mock.patch.object(rollouts, "parse_schema", side_effect=AssertionError("parse_schema should not run")),
            ):
                return await rollouts.generate_math_rollout_rc(
                    cfg,
                    llm,
                    {
                        "task": "2+2?",
                        "answer": "\\boxed{4}",
                        "dataset": "gsm8k_test",
                        "id": 0,
                        "schema": None,
                    },
                    session=None,
                )

        result = asyncio.run(invoke())
        self.assertTrue(result.metrics.success)
        self.assertEqual(result.metrics.reward, 1.0)


if __name__ == "__main__":
    unittest.main()
