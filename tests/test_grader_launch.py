import types
import unittest
from omegaconf import OmegaConf

import pipelinerl.grader_launch as grader_launch
import pipelinerl.launch as launch


class GraderLaunchTest(unittest.TestCase):
    def test_parse_grader_vllm_kwargs_serializes_structured_values(self):
        reserved, extra_args = grader_launch._parse_grader_vllm_kwargs(
            {
                "num_nodes": 2,
                "tensor-parallel-size": 8,
                "enable-expert-parallel": "",
                "disable-cascade-attn": "",
                "reasoning-parser": "step3p5",
                "enable-auto-tool-choice": "",
                "tool-call-parser": "step3p5",
                "hf-overrides": {"num_nextn_predict_layers": 1},
                "speculative_config": {
                    "method": "step3p5_mtp",
                    "num_speculative_tokens": 1,
                },
                "trust-remote-code": "",
            }
        )

        self.assertEqual(reserved["num-nodes"], 2)
        self.assertEqual(reserved["tensor-parallel-size"], 8)
        self.assertEqual(
            extra_args,
            [
                "--enable-expert-parallel",
                "--disable-cascade-attn",
                "--reasoning-parser",
                "step3p5",
                "--enable-auto-tool-choice",
                "--tool-call-parser",
                "step3p5",
                "--hf-overrides",
                '{"num_nextn_predict_layers":1}',
                "--speculative-config",
                '{"method":"step3p5_mtp","num_speculative_tokens":1}',
                "--trust-remote-code",
            ],
        )

    def test_parse_grader_vllm_kwargs_rejects_duplicate_normalized_keys(self):
        with self.assertRaisesRegex(ValueError, "normalize"):
            grader_launch._parse_grader_vllm_kwargs(
                {
                    "speculative_config": {"method": "step3p5_mtp"},
                    "speculative-config": {"method": "other"},
                }
            )

    def test_parse_grader_vllm_kwargs_rejects_blocked_passthrough_key(self):
        with self.assertRaisesRegex(ValueError, "--port"):
            grader_launch._parse_grader_vllm_kwargs({"port": 9000})

    def test_start_llm_grader_builds_sbatch_command_and_uses_vllm_port(self):
        recorded_cmds: list[list[str]] = []
        health_checks: list[tuple[str, int, int]] = []

        def fake_run(cmd, capture_output, text, check):
            recorded_cmds.append(cmd)
            return types.SimpleNamespace(stdout="12345;cluster\n")

        with (
            unittest.mock.patch.object(grader_launch.subprocess, "run", side_effect=fake_run),
            unittest.mock.patch.object(
                grader_launch,
                "_wait_for_slurm_nodes",
                side_effect=lambda job_id, timeout=900: "node-a",
            ),
            unittest.mock.patch.object(grader_launch, "_expand_slurm_node_list", side_effect=lambda nodes: ["node-a"]),
            unittest.mock.patch.object(
                grader_launch,
                "_wait_for_vllm_health",
                side_effect=lambda url, retries, delay, timeout=5: health_checks.append((url, retries, delay)),
            ),
            unittest.mock.patch.object(grader_launch, "_ensure_grader_cleanup_hooks", side_effect=lambda: None),
            unittest.mock.patch.object(grader_launch, "_GRADER_JOB_ID", None),
            unittest.mock.patch.dict(
                grader_launch.os.environ,
                {"HEALTH_CHECK_RETRIES": "7", "HEALTH_CHECK_DELAY": "3"},
                clear=False,
            ),
        ):
            grader_launch.os.environ.pop("SLURM_JOB_ID", None)
            grader_launch.os.environ.pop("OPENAI_BASE_URL", None)
            grader_launch.os.environ.pop("OPENAI_API_KEY", None)

            grader_launch.start_llm_grader(
                "stepfun-ai/Step-3.5-Flash",
                vllm_kwargs={
                    "num_nodes": 2,
                    "data-parallel-size": 4,
                    "tensor-parallel-size": 8,
                    "vllm-port": 8012,
                    "enable-expert-parallel": "",
                    "hf-overrides": {"num_nextn_predict_layers": 1},
                    "speculative-config": {
                        "method": "step3p5_mtp",
                        "num_speculative_tokens": 1,
                    },
                    "trust-remote-code": "",
                },
            )

            self.assertEqual(grader_launch._GRADER_JOB_ID, "12345")
            self.assertEqual(grader_launch.os.environ["OPENAI_BASE_URL"], "http://node-a:8012/v1")
            self.assertEqual(grader_launch.os.environ["OPENAI_API_KEY"], "grader")

        self.assertEqual(
            recorded_cmds,
            [
                [
                    "sbatch",
                    "--parsable",
                    "--nodes=2",
                    "run_grader.slurm",
                    "--model",
                    "stepfun-ai/Step-3.5-Flash",
                    "--data-parallel-size",
                    "4",
                    "--tensor-parallel-size",
                    "8",
                    "--ray-port",
                    "6379",
                    "--vllm-port",
                    "8012",
                    "--max-num-batched-tokens",
                    "8192",
                    "--max-num-seqs",
                    "16",
                    "--max-model-len",
                    "32768",
                    "--gpu-memory-utilization",
                    "0.85",
                    "--",
                    "--api-server-count",
                    "1",
                    "--enable-expert-parallel",
                    "--hf-overrides",
                    '{"num_nextn_predict_layers":1}',
                    "--speculative-config",
                    '{"method":"step3p5_mtp","num_speculative_tokens":1}',
                    "--trust-remote-code",
                ]
            ],
        )
        self.assertEqual(health_checks, [("http://node-a:8012/health", 7, 3)])

    def test_start_llm_grader_omits_passthrough_separator_for_reserved_only(self):
        recorded_cmds: list[list[str]] = []

        def fake_run(cmd, capture_output, text, check):
            recorded_cmds.append(cmd)
            return types.SimpleNamespace(stdout="12345\n")

        with (
            unittest.mock.patch.object(grader_launch.subprocess, "run", side_effect=fake_run),
            unittest.mock.patch.object(
                grader_launch,
                "_wait_for_slurm_nodes",
                side_effect=lambda job_id, timeout=900: "node-a",
            ),
            unittest.mock.patch.object(grader_launch, "_expand_slurm_node_list", side_effect=lambda nodes: ["node-a"]),
            unittest.mock.patch.object(
                grader_launch,
                "_wait_for_vllm_health",
                side_effect=lambda url, retries, delay, timeout=5: None,
            ),
            unittest.mock.patch.object(grader_launch, "_ensure_grader_cleanup_hooks", side_effect=lambda: None),
            unittest.mock.patch.object(grader_launch, "_GRADER_JOB_ID", None),
        ):
            grader_launch.os.environ.pop("SLURM_JOB_ID", None)

            grader_launch.start_llm_grader(
                "openai/gpt-oss-20b",
                vllm_kwargs={"num_nodes": 1, "data-parallel-size": 8, "tensor-parallel-size": 1},
            )

        self.assertEqual(
            recorded_cmds,
            [
                [
                    "sbatch",
                    "--parsable",
                    "--nodes=1",
                    "run_grader.slurm",
                    "--model",
                    "openai/gpt-oss-20b",
                    "--data-parallel-size",
                    "8",
                    "--tensor-parallel-size",
                    "1",
                    "--ray-port",
                    "6379",
                    "--vllm-port",
                    "8000",
                    "--max-num-batched-tokens",
                    "8192",
                    "--max-num-seqs",
                    "16",
                    "--max-model-len",
                    "32768",
                    "--gpu-memory-utilization",
                    "0.85",
                    "--",
                    "--api-server-count",
                    "1",
                ]
            ],
        )

    def test_start_llm_grader_preserves_explicit_api_server_count(self):
        recorded_cmds: list[list[str]] = []

        def fake_run(cmd, capture_output, text, check):
            recorded_cmds.append(cmd)
            return types.SimpleNamespace(stdout="12345\n")

        with (
            unittest.mock.patch.object(grader_launch.subprocess, "run", side_effect=fake_run),
            unittest.mock.patch.object(
                grader_launch,
                "_wait_for_slurm_nodes",
                side_effect=lambda job_id, timeout=900: "node-a",
            ),
            unittest.mock.patch.object(grader_launch, "_expand_slurm_node_list", side_effect=lambda nodes: ["node-a"]),
            unittest.mock.patch.object(
                grader_launch,
                "_wait_for_vllm_health",
                side_effect=lambda url, retries, delay, timeout=5: None,
            ),
            unittest.mock.patch.object(grader_launch, "_ensure_grader_cleanup_hooks", side_effect=lambda: None),
            unittest.mock.patch.object(grader_launch, "_GRADER_JOB_ID", None),
        ):
            grader_launch.os.environ.pop("SLURM_JOB_ID", None)

            grader_launch.start_llm_grader(
                "openai/gpt-oss-20b",
                vllm_kwargs={
                    "num_nodes": 1,
                    "data-parallel-size": 8,
                    "tensor-parallel-size": 1,
                    "api-server-count": 2,
                },
            )

        self.assertEqual(
            recorded_cmds,
            [
                [
                    "sbatch",
                    "--parsable",
                    "--nodes=1",
                    "run_grader.slurm",
                    "--model",
                    "openai/gpt-oss-20b",
                    "--data-parallel-size",
                    "8",
                    "--tensor-parallel-size",
                    "1",
                    "--ray-port",
                    "6379",
                    "--vllm-port",
                    "8000",
                    "--max-num-batched-tokens",
                    "8192",
                    "--max-num-seqs",
                    "16",
                    "--max-model-len",
                    "32768",
                    "--gpu-memory-utilization",
                    "0.85",
                    "--",
                    "--api-server-count",
                    "2",
                ]
            ],
        )


def test_maybe_start_llm_grader_skips_when_name_missing(monkeypatch):
    start_calls: list[tuple[str, object | None]] = []

    monkeypatch.setattr(
        launch,
        "start_llm_grader",
        lambda name, vllm_kwargs=None: start_calls.append((name, vllm_kwargs)),
    )

    cfg = OmegaConf.create(
        {
            "llm_grader": {
                "name": None,
                "vllm_kwargs": {"num_nodes": 1},
            }
        }
    )

    launch._maybe_start_llm_grader(cfg, rank=0)

    assert start_calls == []
