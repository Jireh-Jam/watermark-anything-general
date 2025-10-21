import torch
from watermark_anything.modules.ecc import RepetitionECC, LearnedECCDecoder, ber_from_bits
from watermark_anything.attacks.learned_remover import LearnedRemover


def test_ecc_repetition_basic():
    ecc = RepetitionECC(rep=3)
    msgs = torch.tensor([[0, 1, 1], [1, 0, 1]])
    enc = ecc.encode(msgs)
    assert enc.shape == (2, 3 * 3)
    dec = ecc.decode_majority_from_bits(enc)
    assert torch.equal(dec, msgs.float())


def test_ecc_decode_from_logits():
    ecc = RepetitionECC(rep=3)
    msgs = torch.tensor([[0, 1, 1]])
    enc = ecc.encode(msgs)
    # simulate logits: 0 -> -10, 1 -> +10
    logits = (enc * 20.0) - 10.0
    dec = ecc.decode_majority_from_logits(logits)
    assert torch.equal(dec, msgs.float())


def test_learned_decoder_forward():
    k_raw = 4
    rep = 3
    decoder = LearnedECCDecoder(k_raw=k_raw, rep=rep)
    batch = 2
    dummy_enc = torch.randn(batch, k_raw * rep)
    out = decoder(dummy_enc)
    assert out.shape == (batch, k_raw)


def test_learned_remover_shape_and_scale():
    model = LearnedRemover(in_channels=3, base_ch=16, depth=3, tanh_scale=0.05)
    x = torch.randn(2, 3, 128, 128)
    delta = model(x)
    assert delta.shape == x.shape
    # check magnitude is not huge (tanh_scale ensures small)
    assert float(delta.abs().max()) <= 0.2
