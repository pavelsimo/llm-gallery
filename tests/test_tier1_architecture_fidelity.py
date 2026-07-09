from __future__ import annotations

from dataclasses import asdict

import pytest

from llm_gallery.models import registry

AUDITED_REFERENCE_CONFIGS = {
    "gpt2-xl": {
        "vocab_size": 50257,
        "context_length": 1024,
        "n_layer": 48,
        "n_head": 25,
        "n_embd": 1600,
        "bias": True,
    },
    "llama3-8b": {
        "vocab_size": 128256,
        "context_length": 8192,
        "n_layer": 32,
        "n_head": 32,
        "n_kv_head": 8,
        "n_embd": 4096,
        "intermediate_size": 14336,
        "rope_theta": 500000.0,
        "tie_embeddings": False,
    },
    "deepseek-v3": {
        "vocab_size": 129280,
        "context_length": 163840,
        "n_layer": 61,
        "n_embd": 7168,
        "n_head": 128,
        "q_lora_rank": 1536,
        "kv_lora_rank": 512,
        "qk_nope_head_dim": 128,
        "qk_rope_head_dim": 64,
        "v_head_dim": 128,
        "n_experts": 256,
        "n_experts_per_tok": 8,
        "n_shared_experts": 1,
        "first_k_dense": 3,
    },
    "deepseek-r1": {
        "vocab_size": 129280,
        "context_length": 163840,
        "n_layer": 61,
        "n_embd": 7168,
        "n_head": 128,
        "q_lora_rank": 1536,
        "kv_lora_rank": 512,
        "qk_nope_head_dim": 128,
        "qk_rope_head_dim": 64,
        "v_head_dim": 128,
        "n_experts": 256,
        "n_experts_per_tok": 8,
        "n_shared_experts": 1,
        "first_k_dense": 3,
    },
    "gemma3-27b": {
        "vocab_size": 262208,
        "context_length": 131072,
        "n_layer": 62,
        "n_head": 32,
        "n_kv_head": 16,
        "n_embd": 5376,
        "head_dim": 128,
        "intermediate_size": 21504,
        "sliding_window": 1024,
        "global_every": 6,
        "query_pre_attn_scalar": 168,
    },
    "qwen3-next-80b-a3b": {
        "vocab_size": 151936,
        "context_length": 262144,
        "n_layer": 48,
        "n_embd": 2048,
        "linear_n_head": 16,
        "n_head": 16,
        "n_kv_head": 2,
        "head_dim": 256,
        "partial_rotary_factor": 0.25,
        "attn_every": 4,
        "linear_conv_kernel_dim": 4,
        "n_experts": 512,
        "n_experts_per_tok": 10,
        "n_shared_experts": 1,
    },
    "kimi-linear": {
        "vocab_size": 163840,
        "context_length": 1000000,
        "n_layer": 27,
        "n_embd": 2304,
        "linear_n_head": 32,
        "n_head": 32,
        "n_kv_head": 32,
        "head_dim": 72,
        "attn_every": 4,
        "n_experts": 256,
        "n_experts_per_tok": 8,
        "n_shared_experts": 1,
        "first_k_dense": 1,
        "intermediate_size": 9216,
        "routed_scaling_factor": 2.446,
    },
    "xlstm-7b": {
        "vocab_size": 50304,
        "context_length": 8192,
        "n_layer": 32,
        "n_embd": 4096,
        "n_head": 8,
        "qk_dim_factor": 0.5,
        "gate_soft_cap": 15.0,
        "intermediate_size": 10944,
        "tie_embeddings": False,
    },
    "nemotron3-nano-30b": {
        "vocab_size": 131072,
        "context_length": 262144,
        "n_layer": 52,
        "n_embd": 2688,
        "n_head": 32,
        "n_kv_head": 2,
        "head_dim": 128,
        "layer_pattern": "MEMEM*EMEMEM*EMEMEM*EMEMEM*EMEMEM*EMEMEMEM*EMEMEMEME",
        "mamba_n_heads": 64,
        "mamba_head_dim": 64,
        "mamba_n_groups": 8,
        "mamba_d_state": 128,
        "n_experts": 128,
        "n_experts_per_tok": 6,
        "n_shared_experts": 1,
        "shared_expert_intermediate_size": 3712,
        "routed_scaling_factor": 2.5,
    },
}


def tiny_model(slug: str):
    mod = registry.load(slug)
    cfg = mod.PRESETS["tiny"]
    return mod, cfg, mod.Model(cfg)


@pytest.mark.parametrize("slug, expected", AUDITED_REFERENCE_CONFIGS.items())
def test_audited_reference_configs_match_published_fields(slug: str, expected: dict[str, object]):
    mod = registry.load(slug)
    got = asdict(mod.PRESETS[slug])
    for field, value in expected.items():
        assert got[field] == value


@pytest.mark.parametrize("slug", AUDITED_REFERENCE_CONFIGS)
def test_audited_models_have_tech_report_links(slug: str):
    mod = registry.load(slug)
    assert mod.TECH_REPORT_URL
    assert mod.TECH_REPORT_URL.startswith("https://")


def test_gpt2_matches_classic_mha_absolute_position_stack():
    _, cfg, model = tiny_model("gpt2-xl")
    block = model.blocks[0]

    assert block.attn.n_head == cfg.n_head
    assert block.attn.c_attn.out_features == 3 * cfg.n_embd
    assert block.mlp.c_fc.out_features == 4 * cfg.n_embd
    assert model.wpe.num_embeddings == cfg.context_length
    assert model.lm_head.weight is model.wte.weight
    assert block.ln_1.__class__.__name__ == "LayerNorm"


def test_llama3_matches_gqa_rope_rmsnorm_swiglu_stack():
    _, cfg, model = tiny_model("llama3-8b")
    block = model.blocks[0]

    assert block.attn.n_head == cfg.n_head
    assert block.attn.n_kv_head == cfg.n_kv_head
    assert block.attn.n_rep == cfg.n_head // cfg.n_kv_head
    assert block.attn.q_proj.bias is None
    assert block.ffn.__class__.__name__ == "SwiGLU"
    assert block.attn_norm.__class__.__name__ == "RMSNorm"
    assert hasattr(model, "rope_cos")
    assert model.lm_head.weight is not model.tok_emb.weight


def test_deepseek_v3_and_r1_match_mla_moe_stack():
    for slug in ("deepseek-v3", "deepseek-r1"):
        _, cfg, model = tiny_model(slug)

        assert model.blocks[0].attn.__class__.__name__ == "MLA"
        assert model.blocks[0].ffn.__class__.__name__ == "MLP"
        assert model.blocks[cfg.first_k_dense].ffn.__class__.__name__ == "DeepseekMoE"
        assert model.blocks[cfg.first_k_dense].ffn.top_k == cfg.n_experts_per_tok
        assert model.blocks[cfg.first_k_dense].ffn.shared is not None
        assert model.blocks[0].attn.q_a_proj.out_features == cfg.q_lora_rank
        assert model.blocks[0].attn.kv_a_proj.out_features == (
            cfg.kv_lora_rank + cfg.qk_rope_head_dim
        )


def test_gemma3_matches_sliding_global_sandwich_qk_norm_stack():
    _, cfg, model = tiny_model("gemma3-27b")
    block = model.blocks[0]

    assert [item.is_global for item in model.blocks] == [False, False, True, False, False, True]
    assert block.attn.q_norm.__class__.__name__ == "GemmaRMSNorm"
    assert block.attn.k_norm.__class__.__name__ == "GemmaRMSNorm"
    assert block.post_attn_norm.__class__.__name__ == "GemmaRMSNorm"
    assert block.post_ffn_norm.__class__.__name__ == "GemmaRMSNorm"
    assert block.ffn.__class__.__name__ == "GeGLU"
    assert block.attn.scale == cfg.query_pre_attn_scalar**-0.5
    assert model.lm_head.weight is model.tok_emb.weight


def test_qwen3_next_matches_gated_deltanet_attention_moe_hybrid():
    _, cfg, model = tiny_model("qwen3-next-80b-a3b")

    assert [block.is_attn for block in model.blocks] == [False, False, True, False, False, True]
    assert model.blocks[0].mix.__class__.__name__ == "GatedDeltaNet"
    assert model.blocks[0].mix.A_log.shape == (cfg.linear_n_head,)
    assert model.blocks[2].mix.__class__.__name__ == "GatedAttention"
    assert model.blocks[2].mix.q_norm.__class__.__name__ == "RMSNorm"
    assert model.blocks[0].mix.q_conv.groups == cfg.n_embd
    assert model.blocks[2].mix.rotary_dim == int(cfg.head_dim * cfg.partial_rotary_factor)
    assert model.blocks[0].ffn.__class__.__name__ == "MoE"
    assert model.blocks[0].ffn.top_k == cfg.n_experts_per_tok
    assert model.blocks[0].ffn.shared_gate is not None


def test_kimi_linear_documents_simplified_linear_full_attention_hybrid():
    mod, cfg, model = tiny_model("kimi-linear")

    assert [block.is_attn for block in model.blocks] == [False, False, True, False, False, True]
    assert model.blocks[0].mix.__class__.__name__ == "LinearAttention"
    assert model.blocks[2].mix.__class__.__name__ == "Attention"
    assert model.blocks[0].ffn.__class__.__name__ == "MLP"  # first_k_dense: layer 0 is dense
    assert model.blocks[1].ffn.__class__.__name__ == "MoE"
    assert model.blocks[1].ffn.top_k == cfg.n_experts_per_tok
    assert model.blocks[1].ffn.scaling == cfg.routed_scaling_factor
    assert "ASSUMPTION" in (mod.__doc__ or "")
    assert "plain GQA" in (mod.__doc__ or "")


def test_xlstm_matches_attention_free_recurrent_mlstm_stack():
    _, _, model = tiny_model("xlstm-7b")
    block = model.blocks[0]

    assert block.mix.__class__.__name__ == "mLSTM"
    assert hasattr(block.mix, "i_gate")
    assert hasattr(block.mix, "f_gate")
    assert hasattr(block.mix, "o_gate")
    assert hasattr(block.mix, "out_norm")
    assert block.mix.qk_head_dim == block.mix.head_dim // 2  # qk_dim_factor 0.5
    assert block.ffn.__class__.__name__ == "SwiGLU"
    assert not hasattr(model, "rope_cos")
    assert not hasattr(model, "causal_mask")


def test_nemotron3_nano_matches_mamba_attention_moe_macro_pattern():
    _, cfg, model = tiny_model("nemotron3-nano-30b")

    assert [block.layer_type for block in model.blocks] == list(cfg.layer_pattern)
    mamba = model.blocks[0].mix
    assert mamba.__class__.__name__ == "Mamba"
    assert mamba.d_inner == cfg.mamba_n_heads * cfg.mamba_head_dim
    assert mamba.conv1d.groups == mamba.conv_dim  # conv runs over [x, B, C] (Mamba-2)
    assert mamba.A_log.shape == (cfg.mamba_n_heads,)  # scalar decay per head
    assert mamba.norm.__class__.__name__ == "RMSNorm"  # gated norm before out_proj
    assert model.blocks[1].ffn.__class__.__name__ == "MoE"
    assert model.blocks[1].ffn.top_k == cfg.n_experts_per_tok
    assert not hasattr(model.blocks[1].ffn.experts[0], "gate_proj")  # relu2 MLP, no gate
    assert model.blocks[2].attn.__class__.__name__ == "Attention"
    assert model.blocks[2].attn.n_kv_head == cfg.n_kv_head
