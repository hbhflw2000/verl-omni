"""CPU contract tests; no claim of real-checkpoint or distributed parity."""

import copy
from types import SimpleNamespace

import pytest
import torch
from codec0_reference import (
    align_codec0_logits,
    codec0_forward,
    configure_codec0_policy,
)
from torch import nn


class TinyBackbone(nn.Module):
    def __init__(self):
        super().__init__()
        self.embedding = nn.Embedding(9, 6)
        self.rnn = nn.GRU(6, 6, batch_first=True)

    def forward(self, inputs_embeds, use_cache=False):
        hidden, _ = self.rnn(inputs_embeds)
        return SimpleNamespace(last_hidden_state=hidden)


class Residual(nn.Module):
    def __init__(self):
        super().__init__()
        self.embeddings = nn.ModuleList([nn.Embedding(5, 6) for _ in range(15)])
        self.lm_head = nn.Linear(6, 5)

    def get_input_embeddings(self):
        return self.embeddings

    def forward(self, *args, **kwargs):
        raise AssertionError("Residual predictor forward must not run in codec-0 actor")


class TinyTalker(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = TinyBackbone()
        self.codec_head = nn.Linear(6, 9)
        self.code_predictor = Residual()
        self.text_projection = nn.Linear(6, 6)
        self.config = SimpleNamespace(
            num_code_groups=16,
            code_predictor_config=SimpleNamespace(vocab_size=5),
            codec_eos_token_id=7,
        )

    def get_input_embeddings(self):
        return self.model.embedding


def fixture(frames):
    torch.manual_seed(13)
    model = TinyTalker().double()
    configure_codec0_policy(model)
    return (
        model,
        torch.randn(1, 3, 6, dtype=torch.double),
        torch.randn(1, 1, 6, dtype=torch.double),
        torch.randn(1, 1, 6, dtype=torch.double),
        torch.randint(0, 5, (frames, 16)),
    )


def serial_reference(model, prefix, trailing, pad, codes, eos):
    # Recompute each causal prefix independently as an alignment/gradient oracle.
    inputs = prefix
    rows = []
    actions = len(codes) + int(eos)
    for t in range(actions):
        rows.append(model.codec_head(model.model(inputs).last_hidden_state[:, -1])[0])
        if t + 1 < actions:
            frame = model.get_input_embeddings()(codes[t, 0])
            for j, layer in enumerate(model.code_predictor.get_input_embeddings()):
                frame = frame + layer(codes[t, j + 1])
            inputs = torch.cat(
                (
                    inputs,
                    frame[None, None]
                    + (trailing[:, t : t + 1] if t < trailing.shape[1] else pad),
                ),
                dim=1,
            )
    return torch.stack(rows)


@pytest.mark.parametrize(
    "frames,eos", [(0, True), (1, True), (3, True), (1, False), (3, False)]
)
def test_causal_logits_labels_gradients_and_frozen_parameters(frames, eos):
    model, prefix, trailing, pad, codes = fixture(frames)
    reference = copy.deepcopy(model)
    logits, labels = codec0_forward(
        model, prefix, trailing, pad, codes, terminal_eos=eos
    )
    expected = serial_reference(reference, prefix, trailing, pad, codes, eos)
    allowed = [0, 1, 2, 3, 4, 7]
    torch.testing.assert_close(
        logits[:, allowed], expected[:, allowed], atol=1e-12, rtol=1e-12
    )
    assert labels.tolist() == codes[:, 0].tolist() + ([7] if eos else [])
    assert len(labels) == frames + int(eos)
    assert torch.isneginf(logits[:, [5, 6, 8]]).all()
    loss = -logits.log_softmax(-1).gather(-1, labels[:, None]).sum()
    expected[:, [5, 6, 8]] = -torch.inf
    reference_loss = -expected.log_softmax(-1).gather(-1, labels[:, None]).sum()
    loss.backward()
    reference_loss.backward()
    for (name, p), (_, q) in zip(
        model.named_parameters(), reference.named_parameters(), strict=True
    ):
        if name.startswith(("code_predictor.", "text_projection.")):
            assert not p.requires_grad and p.grad is None
        elif p.grad is not None:
            torch.testing.assert_close(p.grad, q.grad, atol=1e-12, rtol=1e-12)
    assert model.codec_head.weight.grad.abs().sum() > 0
    assert model.model.rnn.weight_ih_l0.grad.abs().sum() > 0
    before = {n: p.detach().clone() for n, p in model.named_parameters()}
    torch.optim.SGD([p for p in model.parameters() if p.requires_grad], lr=0.01).step()
    assert not torch.equal(before["codec_head.weight"], model.codec_head.weight)
    for name, p in model.named_parameters():
        if not p.requires_grad:
            assert torch.equal(before[name], p)


def test_residual_codes_condition_next_frame_but_are_not_actions():
    model, prefix, trailing, pad, codes = fixture(2)
    first, labels = codec0_forward(
        model, prefix, trailing, pad, codes, terminal_eos=True
    )
    changed = codes.clone()
    changed[0, 3] = (changed[0, 3] + 1) % 5
    second, next_labels = codec0_forward(
        model, prefix, trailing, pad, changed, terminal_eos=True
    )
    assert torch.equal(labels, next_labels)
    assert torch.equal(first[0], second[0])
    assert not torch.equal(first[1], second[1])


@pytest.mark.parametrize(
    "fault",
    [
        "missing_group",
        "bad_id",
        "float_codes",
        "empty_cutoff",
        "nonbool_eos",
        "trainable_residual",
        "nan_prefix",
        "bad_eos",
    ],
)
def test_invalid_contract_rejected(fault):
    model, prefix, trailing, pad, codes = fixture(2)
    eos = True
    if fault == "missing_group":
        codes = codes[:, :-1]
    if fault == "bad_id":
        codes[0, 4] = 5
    if fault == "float_codes":
        codes = codes.float()
    if fault == "empty_cutoff":
        codes, eos = codes[:0], False
    if fault == "nonbool_eos":
        eos = 1
    if fault == "trainable_residual":
        model.code_predictor.requires_grad_(True)
    if fault == "nan_prefix":
        prefix[0, 0, 0] = float("nan")
    if fault == "bad_eos":
        model.config.codec_eos_token_id = 10
    with pytest.raises(ValueError):
        codec0_forward(model, prefix, trailing, pad, codes, terminal_eos=eos)


def test_next_token_alignment_padding_entropy():
    model, prefix, trailing, pad, codes = fixture(2)
    logits, labels = codec0_forward(
        model, prefix, trailing, pad, codes, terminal_eos=True
    )
    ids = torch.cat((torch.tensor([0, 0]), labels, torch.tensor([151643])))
    mask = torch.tensor([1, 1, 1, 1, 1, 0])
    aligned = align_codec0_logits(logits, labels, ids, mask)
    lp = aligned.log_softmax(-1)
    torch.testing.assert_close(
        lp[1:4].gather(-1, labels[:, None]),
        logits.float().log_softmax(-1).gather(-1, labels[:, None]),
    )
    assert torch.isfinite(-(aligned.softmax(-1) * lp).sum(-1)).all()
    assert aligned[-1].eq(0).all()


@pytest.mark.parametrize(
    "ids,mask",
    [
        ([0, 2, 1], [1, 1, 1]),
        ([1, 2], [1, 1]),
        ([0, 1, 2], [1, 0, 1]),
        ([0, 1, 2], [1, 0.5, 1]),
        ([9, 1, 2], [1, 1, 1]),
    ],
)
def test_bad_response_alignment(ids, mask):
    with pytest.raises(ValueError):
        align_codec0_logits(
            torch.zeros(2, 4),
            torch.tensor([1, 2]),
            torch.tensor(ids),
            torch.tensor(mask),
        )
