# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Unit tests for the DGX Spark (GB10 / SM121) NemotronH default folding.

These exercise ``NemotronHForCausalLMConfig.apply_dgx_spark_defaults`` in
isolation: the folding must apply only on Spark for an NVFP4 checkpoint, only
touch flags left at their default sentinel, and never override an explicit
value.
"""

from types import SimpleNamespace

import pytest

from vllm.config.mamba import MambaBackendEnum
from vllm.model_executor.models.config import NemotronHForCausalLMConfig


def _make_vllm_config(quant: str | None = "modelopt_mixed") -> SimpleNamespace:
    """Build a stand-in VllmConfig with every folded flag at its default."""
    return SimpleNamespace(
        model_config=SimpleNamespace(quantization=quant),
        cache_config=SimpleNamespace(mamba_ssm_cache_dtype="auto"),
        mamba_config=SimpleNamespace(
            backend=MambaBackendEnum.TRITON,
            enable_stochastic_rounding=False,
            stochastic_rounding_philox_rounds=0,
        ),
        kernel_config=SimpleNamespace(
            moe_backend="auto",
            linear_backend="auto",
            enable_flashinfer_autotune=None,
        ),
        scheduler_config=SimpleNamespace(async_scheduling=None),
        compilation_config=SimpleNamespace(
            pass_config=SimpleNamespace(fuse_norm_quant=None, fuse_act_quant=None)
        ),
    )


@pytest.fixture
def on_spark(monkeypatch):
    from vllm.platforms import current_platform

    monkeypatch.setattr(
        current_platform,
        "is_device_capability",
        lambda capability, device_id=0: capability == (12, 1),
    )


def test_folds_defaults_on_spark_nvfp4(on_spark):
    cfg = _make_vllm_config()
    NemotronHForCausalLMConfig.apply_dgx_spark_defaults(cfg)

    assert cfg.cache_config.mamba_ssm_cache_dtype == "float16"
    assert cfg.mamba_config.backend == MambaBackendEnum.FLASHINFER
    assert cfg.mamba_config.enable_stochastic_rounding is True
    assert cfg.mamba_config.stochastic_rounding_philox_rounds == 5
    assert cfg.kernel_config.moe_backend == "flashinfer_b12x"
    assert cfg.kernel_config.linear_backend == "flashinfer_cutlass"
    assert cfg.kernel_config.enable_flashinfer_autotune is True
    assert cfg.scheduler_config.async_scheduling is True
    assert cfg.compilation_config.pass_config.fuse_norm_quant is True
    assert cfg.compilation_config.pass_config.fuse_act_quant is True


def test_explicit_values_are_preserved(on_spark):
    cfg = _make_vllm_config()
    cfg.kernel_config.moe_backend = "marlin"
    cfg.mamba_config.backend = MambaBackendEnum.FLASHINFER
    cfg.scheduler_config.async_scheduling = False

    NemotronHForCausalLMConfig.apply_dgx_spark_defaults(cfg)

    assert cfg.kernel_config.moe_backend == "marlin"
    assert cfg.mamba_config.backend == MambaBackendEnum.FLASHINFER
    assert cfg.scheduler_config.async_scheduling is False


def test_noop_when_not_nvfp4(on_spark):
    cfg = _make_vllm_config(quant="awq")
    NemotronHForCausalLMConfig.apply_dgx_spark_defaults(cfg)

    assert cfg.kernel_config.moe_backend == "auto"
    assert cfg.mamba_config.backend == MambaBackendEnum.TRITON


def test_noop_off_spark(monkeypatch):
    from vllm.platforms import current_platform

    # SM120 (RTX 50-series) is Blackwell but not DGX Spark.
    monkeypatch.setattr(
        current_platform,
        "is_device_capability",
        lambda capability, device_id=0: False,
    )
    cfg = _make_vllm_config()
    NemotronHForCausalLMConfig.apply_dgx_spark_defaults(cfg)

    assert cfg.kernel_config.moe_backend == "auto"
    assert cfg.cache_config.mamba_ssm_cache_dtype == "auto"
