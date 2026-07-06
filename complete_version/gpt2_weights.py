"""Helpers for loading OpenAI GPT-2 checkpoint arrays into GPTModel."""

import numpy as np
import torch


def assign(left, right):
    right = torch.tensor(right)
    if left.shape != right.shape:
        raise ValueError(f"Shape mismatch. Left: {left.shape}, Right: {right.shape}")
    return torch.nn.Parameter(right)


def load_weights_into_gpt(gpt, params):
    gpt.pos_emb.weight = assign(gpt.pos_emb.weight, params["wpe"])
    gpt.tok_emb.weight = assign(gpt.tok_emb.weight, params["wte"])

    for block_idx in range(len(params["blocks"])):
        block = params["blocks"][block_idx]
        q_w, k_w, v_w = np.split(block["attn"]["c_attn"]["w"], 3, axis=-1)
        gpt.trf_blocks[block_idx].att.W_query.weight = assign(
            gpt.trf_blocks[block_idx].att.W_query.weight,
            q_w.T,
        )
        gpt.trf_blocks[block_idx].att.W_key.weight = assign(
            gpt.trf_blocks[block_idx].att.W_key.weight,
            k_w.T,
        )
        gpt.trf_blocks[block_idx].att.W_value.weight = assign(
            gpt.trf_blocks[block_idx].att.W_value.weight,
            v_w.T,
        )

        q_b, k_b, v_b = np.split(block["attn"]["c_attn"]["b"], 3, axis=-1)
        gpt.trf_blocks[block_idx].att.W_query.bias = assign(
            gpt.trf_blocks[block_idx].att.W_query.bias,
            q_b,
        )
        gpt.trf_blocks[block_idx].att.W_key.bias = assign(
            gpt.trf_blocks[block_idx].att.W_key.bias,
            k_b,
        )
        gpt.trf_blocks[block_idx].att.W_value.bias = assign(
            gpt.trf_blocks[block_idx].att.W_value.bias,
            v_b,
        )

        gpt.trf_blocks[block_idx].att.out_proj.weight = assign(
            gpt.trf_blocks[block_idx].att.out_proj.weight,
            block["attn"]["c_proj"]["w"].T,
        )
        gpt.trf_blocks[block_idx].att.out_proj.bias = assign(
            gpt.trf_blocks[block_idx].att.out_proj.bias,
            block["attn"]["c_proj"]["b"],
        )

        gpt.trf_blocks[block_idx].ff.layers[0].weight = assign(
            gpt.trf_blocks[block_idx].ff.layers[0].weight,
            block["mlp"]["c_fc"]["w"].T,
        )
        gpt.trf_blocks[block_idx].ff.layers[0].bias = assign(
            gpt.trf_blocks[block_idx].ff.layers[0].bias,
            block["mlp"]["c_fc"]["b"],
        )
        gpt.trf_blocks[block_idx].ff.layers[2].weight = assign(
            gpt.trf_blocks[block_idx].ff.layers[2].weight,
            block["mlp"]["c_proj"]["w"].T,
        )
        gpt.trf_blocks[block_idx].ff.layers[2].bias = assign(
            gpt.trf_blocks[block_idx].ff.layers[2].bias,
            block["mlp"]["c_proj"]["b"],
        )

        gpt.trf_blocks[block_idx].norm1.scale = assign(
            gpt.trf_blocks[block_idx].norm1.scale,
            block["ln_1"]["g"],
        )
        gpt.trf_blocks[block_idx].norm1.shift = assign(
            gpt.trf_blocks[block_idx].norm1.shift,
            block["ln_1"]["b"],
        )
        gpt.trf_blocks[block_idx].norm2.scale = assign(
            gpt.trf_blocks[block_idx].norm2.scale,
            block["ln_2"]["g"],
        )
        gpt.trf_blocks[block_idx].norm2.shift = assign(
            gpt.trf_blocks[block_idx].norm2.shift,
            block["ln_2"]["b"],
        )

    gpt.final_norm.scale = assign(gpt.final_norm.scale, params["g"])
    gpt.final_norm.shift = assign(gpt.final_norm.shift, params["b"])
    gpt.out_head.weight = assign(gpt.out_head.weight, params["wte"])
