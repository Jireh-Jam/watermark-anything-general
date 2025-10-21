import torch
from watermark_anything.modules.msg_processor_film import MsgProcessorFiLM


def test_msg_processor_film_shapes():
    b = 2
    nbits = 8
    latent_c = 64
    processor = MsgProcessorFiLM(nbits=nbits, latent_channels=latent_c, hidden_size=128)
    latents = torch.randn(b, latent_c, 8, 8)
    msgs = torch.randint(0, 2, (b, nbits)).float()
    out = processor(latents, msgs)
    assert out.shape == latents.shape
    # ensure some change occurs vs identity
    diff = (out - latents).abs().mean()
    assert diff.item() > 1e-6
