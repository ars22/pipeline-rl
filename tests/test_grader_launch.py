import types

import pytest

import pipelinerl.grader_launch as grader_launch


def test_parse_grader_vllm_kwargs_serializes_structured_values():
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

    assert reserved["num-nodes"] == 2
    assert reserved["tensor-parallel-size"] == 8
    assert extra_args == [
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
    ]


def test_parse_grader_vllm_kwargs_rejects_duplicate_normalized_keys():
    with pytest.raises(ValueError, match="normalize"):
        grader_launch._parse_grader_vllm_kwargs(
            {
                "speculative_config": {"method": "step3p5_mtp"},
                "speculative-config": {"method": "other"},
            }
        )


def test_parse_grader_vllm_kwargs_rejects_blocked_passthrough_key():
    with pytest.raises(ValueError, match="--port"):
        grader_launch._parse_grader_vllm_kwargs({"port": 9000})


def test_start_llm_grader_builds_sbatch_command_and_uses_vllm_port(monkeypatch):
    recorded_cmds: list[list[str]] = []
    health_checks: list[tuple[str, int, int]] = []

    def fake_run(cmd, capture_output, text, check):
        recorded_cmds.append(cmd)
        return types.SimpleNamespace(stdout="12345;cluster\n")

    monkeypatch.setattr(grader_launch.subprocess, "run", fake_run)
    monkeypatch.setattr(grader_launch, "_wait_for_slurm_nodes", lambda job_id, timeout=900: "node-a")
    monkeypatch.setattr(grader_launch, "_expand_slurm_node_list", lambda nodes: ["node-a"])
    monkeypatch.setattr(
        grader_launch,
        "_wait_for_vllm_health",
        lambda url, retries, delay, timeout=5: health_checks.append((url, retries, delay)),
    )
    monkeypatch.setattr(grader_launch, "_ensure_grader_cleanup_hooks", lambda: None)
    monkeypatch.setattr(grader_launch, "_GRADER_JOB_ID", None)
    monkeypatch.setenv("HEALTH_CHECK_RETRIES", "7")
    monkeypatch.setenv("HEALTH_CHECK_DELAY", "3")
    monkeypatch.delenv("SLURM_JOB_ID", raising=False)
    monkeypatch.delenv("OPENAI_BASE_URL", raising=False)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)

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

    assert recorded_cmds == [
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
    ]
    assert health_checks == [("http://node-a:8012/health", 7, 3)]
    assert grader_launch._GRADER_JOB_ID == "12345"
    assert grader_launch.os.environ["OPENAI_BASE_URL"] == "http://node-a:8012/v1"
    assert grader_launch.os.environ["OPENAI_API_KEY"] == "grader"


def test_start_llm_grader_omits_passthrough_separator_for_reserved_only(monkeypatch):
    recorded_cmds: list[list[str]] = []

    def fake_run(cmd, capture_output, text, check):
        recorded_cmds.append(cmd)
        return types.SimpleNamespace(stdout="12345\n")

    monkeypatch.setattr(grader_launch.subprocess, "run", fake_run)
    monkeypatch.setattr(grader_launch, "_wait_for_slurm_nodes", lambda job_id, timeout=900: "node-a")
    monkeypatch.setattr(grader_launch, "_expand_slurm_node_list", lambda nodes: ["node-a"])
    monkeypatch.setattr(grader_launch, "_wait_for_vllm_health", lambda url, retries, delay, timeout=5: None)
    monkeypatch.setattr(grader_launch, "_ensure_grader_cleanup_hooks", lambda: None)
    monkeypatch.setattr(grader_launch, "_GRADER_JOB_ID", None)
    monkeypatch.delenv("SLURM_JOB_ID", raising=False)

    grader_launch.start_llm_grader(
        "openai/gpt-oss-20b",
        vllm_kwargs={"num_nodes": 1, "data-parallel-size": 8, "tensor-parallel-size": 1},
    )

    assert "--" not in recorded_cmds[0]


def test_start_llm_grader_preserves_explicit_api_server_count(monkeypatch):
    recorded_cmds: list[list[str]] = []

    def fake_run(cmd, capture_output, text, check):
        recorded_cmds.append(cmd)
        return types.SimpleNamespace(stdout="12345\n")

    monkeypatch.setattr(grader_launch.subprocess, "run", fake_run)
    monkeypatch.setattr(grader_launch, "_wait_for_slurm_nodes", lambda job_id, timeout=900: "node-a")
    monkeypatch.setattr(grader_launch, "_expand_slurm_node_list", lambda nodes: ["node-a"])
    monkeypatch.setattr(grader_launch, "_wait_for_vllm_health", lambda url, retries, delay, timeout=5: None)
    monkeypatch.setattr(grader_launch, "_ensure_grader_cleanup_hooks", lambda: None)
    monkeypatch.setattr(grader_launch, "_GRADER_JOB_ID", None)
    monkeypatch.delenv("SLURM_JOB_ID", raising=False)

    grader_launch.start_llm_grader(
        "openai/gpt-oss-20b",
        vllm_kwargs={
            "num_nodes": 1,
            "data-parallel-size": 8,
            "tensor-parallel-size": 1,
            "api-server-count": 2,
        },
    )

    assert recorded_cmds == [
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
    ]
