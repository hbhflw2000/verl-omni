"""Codec-0 actor math reference for RFC #800; not a registered training pipeline."""

import torch


def configure_codec0_policy(talker):
    """Freeze conditioning projections and the residual predictor before wrapping."""
    for name, parameter in talker.named_parameters():
        parameter.requires_grad_(name.startswith(("model.", "codec_head.")))


def codec0_forward(talker, prefix, trailing, pad, codes, *, terminal_eos):
    """Teacher-force codec-0 actions with recorded residual codes as conditioning.

    The caller constructs the native conditioning tensors, preserving any
    trainable prefix embeddings in the graph. This function does not define a
    rollout payload or implement Thinker conditioning extraction.
    """
    if type(terminal_eos) is not bool:
        raise ValueError("terminal_eos must explicitly distinguish EOS from truncation")
    if prefix.ndim != 3 or prefix.shape[0] != 1 or prefix.shape[1] < 1:
        raise ValueError("Expected one nonempty conditioning prefix")
    width = prefix.shape[-1]
    if trailing.ndim != 3 or trailing.shape[0] != 1 or trailing.shape[-1] != width:
        raise ValueError("Invalid trailing conditioning")
    if pad.shape != (1, 1, width):
        raise ValueError("Invalid padding embedding")
    for tensor in (prefix, trailing, pad):
        if not tensor.is_floating_point() or not bool(torch.isfinite(tensor).all()):
            raise ValueError("Conditioning must be finite floating point")
        if tensor.device != prefix.device or tensor.dtype != prefix.dtype:
            raise ValueError("Conditioning device and dtype must agree")
    config = talker.config
    groups = config.num_code_groups
    vocab = config.code_predictor_config.vocab_size
    if codes.ndim != 2 or codes.shape[1] != groups or codes.dtype != torch.long:
        raise ValueError("Expected int64 frames by codebooks")
    if codes.device != prefix.device or bool(((codes < 0) | (codes >= vocab)).any()):
        raise ValueError("Invalid codec IDs or device")
    if not len(codes) and not terminal_eos:
        raise ValueError("Empty trajectory requires natural EOS")
    residual_embeddings = talker.code_predictor.get_input_embeddings()
    if len(residual_embeddings) != groups - 1:
        raise ValueError("Residual embedding count does not match checkpoint config")
    if any(p.requires_grad for p in talker.code_predictor.parameters()):
        raise ValueError("The residual predictor must be frozen for codec-0 training")
    labels = codes[:, 0]
    if terminal_eos:
        labels = torch.cat((labels, codes.new_tensor([config.codec_eos_token_id])))

    # One frame is consumed only to predict the next action; never score residual IDs.
    consumed = len(labels) - 1
    frames = codes[:consumed]
    if consumed:
        pieces = [talker.get_input_embeddings()(frames[:, 0])[None]]
        pieces.extend(
            layer(frames[:, i + 1])[None] for i, layer in enumerate(residual_embeddings)
        )
        frame_embeddings = torch.stack(pieces, dim=1).sum(dim=1)
        text = torch.cat(
            (
                trailing[:, :consumed],
                pad.expand(1, max(0, consumed - trailing.shape[1]), width),
            ),
            dim=1,
        )
        inputs = torch.cat((prefix, frame_embeddings + text), dim=1)
    else:
        inputs = prefix
    output = talker.model(inputs_embeds=inputs, use_cache=False)
    logits = talker.codec_head(output.last_hidden_state[:, prefix.shape[1] - 1 :])[0]
    eos = config.codec_eos_token_id
    if not 0 < vocab <= logits.shape[-1] or not vocab <= eos < logits.shape[-1]:
        raise ValueError("Invalid checkpoint codec vocabulary/EOS")
    support = torch.arange(logits.shape[-1], device=logits.device) < vocab
    support[eos] = True
    return logits.masked_fill(~support, -torch.inf), labels


def align_codec0_logits(logits, labels, input_ids, attention_mask):
    """Align codec-0 actions and optional natural EOS to next-token positions."""
    if input_ids.ndim != 1 or attention_mask.shape != input_ids.shape:
        raise ValueError("Expected matching one-dimensional actor IDs and mask")
    if (
        logits.ndim != 2
        or labels.ndim != 1
        or len(labels) == 0
        or len(logits) != len(labels)
    ):
        raise ValueError("Expected nonempty action logits and labels")
    if input_ids.dtype != torch.long or labels.dtype != torch.long:
        raise ValueError("IDs and labels must be int64")
    if not bool(((attention_mask == 0) | (attention_mask == 1)).all()):
        raise ValueError("Attention mask must be binary")
    length = int(attention_mask.sum())
    if not torch.equal(
        attention_mask.bool(),
        torch.arange(len(input_ids), device=input_ids.device) < length,
    ):
        raise ValueError("Actor inputs must be right padded without gaps")
    start = length - len(labels)
    if start < 1 or not torch.equal(input_ids[start:length], labels):
        raise ValueError("Codec-0 actions do not match response IDs")
    if bool(
        ((input_ids[:length] < 0) | (input_ids[:length] >= logits.shape[-1])).any()
    ):
        raise ValueError("Unmasked actor IDs exceed the policy vocabulary")
    if bool(torch.isnan(logits).any()) or bool(torch.isposinf(logits).any()):
        raise ValueError("Invalid action logits")
    if not bool(torch.isfinite(logits.gather(-1, labels[:, None])).all()):
        raise ValueError("Action lies outside policy support")
    finite = logits.float().masked_fill(
        torch.isneginf(logits), torch.finfo(torch.float32).min
    )
    aligned = finite.new_zeros((len(input_ids), logits.shape[-1]))
    aligned[start - 1 : length - 1] = finite
    return aligned
