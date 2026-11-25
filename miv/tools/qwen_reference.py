"""Independent NumPy Qwen oracle over represented source weights.

This deliberately does not use native plans, native kernels, or the artifact
decoder. Matrix/normalization/attention calculations use FP64; GDN persistent
state is FP32. BF16 is rounded at the public producer outputs consumed by the
next Op. Fused operations do not insert private MLX intermediate roundings.
"""

from __future__ import annotations

import numpy as np


def bf16(value):
    value = np.asarray(value, dtype=np.float32)
    bits = value.view(np.uint32)
    rounded = (bits + np.uint32(0x7FFF) + ((bits >> 16) & 1)) & np.uint32(0xFFFF0000)
    return rounded.view(np.float32).astype(np.float64)


def rms(value, weight, epsilon):
    value = np.asarray(value, dtype=np.float64)
    return (
        value
        / np.sqrt(np.mean(value * value, axis=-1, keepdims=True) + epsilon)
        * weight
    )


def sigmoid(value):
    return 1 / (1 + np.exp(-value))


def silu(value):
    return value * sigmoid(value)


class QwenReference:
    def __init__(self, config, source, mode, overrides=None, norm_offset=0):
        self.config = config
        self.source = source
        self.mode = mode
        self.overrides = overrides or {}
        self.norm_offset = norm_offset
        self.decoded = {}

    def weight(self, name):
        if name in self.decoded:
            return self.decoded[name]
        packed = self.source[name + ".weight"]
        if name + ".scales" not in self.source:
            return packed.astype(np.float64)
        scales = self.source[name + ".scales"]
        spec = self.overrides.get(name, {"mode": self.mode, "bits": 4})
        bits = spec["bits"]
        # Source Q5/Q6 codes cross uint32 boundaries. Decode the bitstream
        # independently of the native base/high-plane conversion.
        raw = np.ascontiguousarray(packed, dtype="<u4").view(np.uint8)
        fields = np.unpackbits(raw, axis=-1, bitorder="little").reshape(
            *packed.shape[:-1], -1, bits
        )
        codes = np.sum(fields * (1 << np.arange(bits)), axis=-1).astype(np.uint8)
        if spec["mode"] == "affine":
            group = codes.shape[-1] // scales.shape[-1]
            result = codes.astype(np.float64) * np.repeat(scales, group, -1)
            result += np.repeat(self.source[name + ".biases"], group, -1)
        else:
            magnitude = np.array([0, 0.5, 1, 1.5, 2, 3, 4, 6], dtype=np.float64)
            result = magnitude[codes & 7] * np.where(codes & 8, -1, 1)
            result *= np.repeat(np.exp2(scales.astype(np.float64) - 127), 32, -1)
        self.decoded[name] = result
        return result

    def linear(self, x, name):
        return bf16(x @ self.weight(name).T)

    def normalization(self, x, name):
        return bf16(
            rms(x, self.weight(name) + self.norm_offset, self.config["rms_norm_eps"])
        )

    def attention(self, x, name):
        c = self.config
        tokens = len(x)
        heads, kvheads, dim = (
            c["num_attention_heads"],
            c["num_key_value_heads"],
            c["head_dim"],
        )
        qg = self.linear(x, name + ".q_proj").reshape(tokens, heads, 2, dim)
        q, gate = qg[:, :, 0], qg[:, :, 1]
        k = self.linear(x, name + ".k_proj").reshape(tokens, kvheads, dim)
        v = self.linear(x, name + ".v_proj").reshape(tokens, kvheads, dim)
        rope = c["rope_parameters"]
        rotary = round(dim * rope["partial_rotary_factor"])
        angles = np.arange(tokens)[:, None] / (
            rope["rope_theta"] ** (np.arange(rotary // 2) * 2 / rotary)
        )

        def norm_rope(x, weight):
            y = rms(x, self.weight(weight) + self.norm_offset, c["rms_norm_eps"])
            left = y[..., : rotary // 2].copy()
            right = y[..., rotary // 2 : rotary].copy()
            cosine, sine = np.cos(angles)[:, None, :], np.sin(angles)[:, None, :]
            y[..., : rotary // 2] = left * cosine - right * sine
            y[..., rotary // 2 : rotary] = left * sine + right * cosine
            return bf16(y)

        q = norm_rope(q, name + ".q_norm")
        k = norm_rope(k, name + ".k_norm")
        output = np.empty_like(q)
        for t in range(tokens):
            for head in range(heads):
                kv = head // (heads // kvheads)
                score = k[: t + 1, kv] @ q[t, head] / np.sqrt(dim)
                prob = np.exp(score - np.max(score))
                prob /= prob.sum()
                output[t, head] = (prob @ v[: t + 1, kv]) * sigmoid(gate[t, head])
        return self.linear(bf16(output).reshape(tokens, -1), name + ".o_proj")

    def gdn(self, x, name):
        c = self.config
        hk, hv = c["linear_num_key_heads"], c["linear_num_value_heads"]
        dk, dv = c["linear_key_head_dim"], c["linear_value_head_dim"]
        qkv = self.linear(x, name + ".in_proj_qkv")
        z = self.linear(x, name + ".in_proj_z").reshape(len(x), hv, dv)
        a = self.linear(x, name + ".in_proj_a")
        b = self.linear(x, name + ".in_proj_b")
        kernel = c["linear_conv_kernel_dim"]
        weight = self.weight(name + ".conv1d").reshape(qkv.shape[-1], kernel)
        padded = np.pad(qkv, ((kernel - 1, 0), (0, 0)))
        convolution = np.stack(
            [np.sum(padded[t : t + kernel] * weight.T, axis=0) for t in range(len(x))]
        )
        convolved = bf16(silu(convolution))
        q = convolved[:, : hk * dk].reshape(len(x), hk, dk)
        k = convolved[:, hk * dk : 2 * hk * dk].reshape(len(x), hk, dk)
        v = convolved[:, 2 * hk * dk :].reshape(len(x), hv, dv)
        q = rms(q, 1, 1e-6) / dk
        k = rms(k, 1, 1e-6) / np.sqrt(dk)
        state = np.zeros((hv, dv, dk), dtype=np.float32)
        recurrent = np.empty((len(x), hv, dv))
        for t in range(len(x)):
            for head in range(hv):
                key_head = head // (hv // hk)
                activation = a[t, head] + self.source[name + ".dt_bias"][head]
                decay = np.exp(
                    -np.exp(self.source[name + ".A_log"][head])
                    * np.logaddexp(0, activation)
                )
                decayed = (state[head] * decay).astype(np.float32)
                delta = (v[t, head] - decayed @ k[t, key_head]) * sigmoid(b[t, head])
                state[head] = decayed + delta[:, None] * k[t, key_head]
                recurrent[t, head] = state[head] @ q[t, key_head]
        normalized = rms(
            bf16(recurrent), self.weight(name + ".norm"), c["rms_norm_eps"]
        )
        return self.linear(
            bf16(normalized * silu(z)).reshape(len(x), -1), name + ".out_proj"
        )

    def __call__(self, tokens, route_observation=None):
        self.route_history = []
        self.route_observation = route_observation
        self.route_qualification = []
        x = bf16(self.weight("model.embed_tokens")[tokens])
        features = []
        for layer, kind in enumerate(self.config["layer_types"]):
            name = f"model.layers.{layer}"
            normalized = self.normalization(x, name + ".input_layernorm")
            delta = (
                self.attention(normalized, name + ".self_attn")
                if kind == "full_attention"
                else self.gdn(normalized, name + ".linear_attn")
            )
            x = bf16(x + delta)
            normalized = self.normalization(x, name + ".post_attention_layernorm")
            down = self.mlp(normalized, name + ".mlp")
            x = bf16(x + down)
            features.append(x.copy())
        hidden = self.normalization(x, "model.norm")
        return hidden, self.linear(hidden, "lm_head"), np.stack(features, axis=1)

    def mlp(self, x, name):
        if not self.config.get("num_experts", 0):
            gate = self.linear(x, name + ".gate_proj")
            up = self.linear(x, name + ".up_proj")
            return self.linear(bf16(silu(gate) * up), name + ".down_proj")
        router = self.linear(x, name + ".gate")
        observed = self.route_observation
        layer = len(self.route_history)
        if observed is not None:
            actual = np.asarray(observed["router_logits"])[:, layer]
            difference = actual - router
            max_error = float(np.max(np.abs(difference)))
            relative = float(
                np.linalg.norm(difference) / max(np.linalg.norm(router), 1e-12)
            )
            if not np.isfinite(actual).all() or max_error > 0.03125 or relative > 0.005:
                raise AssertionError(
                    f"{name}: router projection max={max_error} relative={relative}"
                )
            reference_routes = np.stack(
                [np.lexsort((-np.arange(256), -row))[:8] for row in router]
            )
            router = actual
            changed = 0
        else:
            reference_routes = None
        expert_name = name + ".switch_mlp"
        gate_bank = self.weight(expert_name + ".gate_proj")
        up_bank = self.weight(expert_name + ".up_proj")
        down_bank = self.weight(expert_name + ".down_proj")
        routed = []
        layer_routes = []
        for token, row in enumerate(x):
            # The public router breaks equal scores by descending expert ID.
            indices = np.lexsort((-np.arange(256), -router[token]))[:8]
            layer_routes.append(indices)
            scores = np.exp(router[token, indices] - np.max(router[token, indices]))
            scores = bf16(scores / scores.sum())
            if observed is not None:
                actual_ids = np.asarray(observed["layer_routes"])[token, layer]
                if not np.array_equal(actual_ids, indices):
                    raise AssertionError(f"{name}: top-8 violates represented logits")
                actual_scores = np.asarray(observed["router_scores"])[token, layer]
                if (
                    not np.isfinite(actual_scores).all()
                    or np.max(abs(actual_scores - scores)) > 0.001
                ):
                    raise AssertionError(f"{name}: normalized expert scores disagree")
                changed += int(set(reference_routes[token]) != set(indices))
                scores = actual_scores
            outputs = []
            for expert, score in zip(indices, scores):
                gate = bf16(gate_bank[expert] @ row)
                up = bf16(up_bank[expert] @ row)
                activation = bf16(silu(gate) * up)
                outputs.append(bf16((down_bank[expert] @ activation) * score))
            routed.append(bf16(np.sum(outputs, axis=0)))
        if observed is not None:
            self.route_qualification.append(
                {
                    "layer": layer,
                    "router_max_abs": max_error,
                    "router_relative_l2": relative,
                    "changed_route_rows": changed,
                }
            )
        self.route_history.append(np.asarray(layer_routes))
        gate = self.linear(x, name + ".shared_expert.gate_proj")
        up = self.linear(x, name + ".shared_expert.up_proj")
        shared = self.linear(bf16(silu(gate) * up), name + ".shared_expert.down_proj")
        scale = sigmoid(self.linear(x, name + ".shared_expert_gate"))
        return bf16(np.asarray(routed) + scale * shared)
