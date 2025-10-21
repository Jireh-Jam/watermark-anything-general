import torch
from torch import nn


class MsgProcessorFiLM(nn.Module):
    """
    FiLM-style message processor.

    - nbits: number of raw message bits
    - latent_channels: number of channels in the encoder latents (C)
    - hidden_size: optional hidden dim for message embedding network
    Usage:
        processor = MsgProcessorFiLM(nbits=16, latent_channels=64)
        latents_w = processor(latents, msgs)
    """

    def __init__(self, nbits: int, latent_channels: int, hidden_size: int = None):
        super().__init__()
        assert nbits >= 0
        self.nbits = nbits
        self.latent_channels = latent_channels
        if hidden_size is None:
            hidden_size = max(128, nbits * 8 if nbits > 0 else 128)

        # message embedding network (maps binary message -> embedding)
        if nbits > 0:
            self.msg_emb = nn.Sequential(
                nn.Linear(nbits, hidden_size),
                nn.ReLU(inplace=True),
                nn.Linear(hidden_size, hidden_size),
                nn.ReLU(inplace=True),
            )
            # produce scale and shift for FiLM (one value per latent channel)
            self.film_scale = nn.Linear(hidden_size, latent_channels)
            self.film_shift = nn.Linear(hidden_size, latent_channels)
        else:
            # no-op if no bits
            self.msg_emb = None
            self.film_scale = None
            self.film_shift = None

    def get_random_msg(self, bsz: int = 1, nb_repetitions: int = 1) -> torch.Tensor:
        """
        convenience: creates random binary messages of shape (bsz, nbits)
        """
        if self.nbits == 0:
            return torch.tensor([], dtype=torch.float32)
        if nb_repetitions != 1:
            assert self.nbits % nb_repetitions == 0
            aux = torch.randint(0, 2, (bsz, self.nbits // nb_repetitions))
            return aux.unsqueeze(1).repeat(1, nb_repetitions, 1).view(bsz, self.nbits)
        return torch.randint(0, 2, (bsz, self.nbits))

    def forward(self, latents: torch.Tensor, msgs: torch.Tensor) -> torch.Tensor:
        """
        latents: B x C x H x W  (or B x C if flattened)
        msgs: B x nbits  (binary 0/1 or floats)
        returns: modulated latents: (1 + scale) * latents + shift
        """
        if self.nbits == 0 or msgs is None or msgs.numel() == 0:
            return latents

        msgs = msgs.to(latents.device).float()
        emb = self.msg_emb(msgs)  # B x hidden
        scale = self.film_scale(emb)  # B x C
        shift = self.film_shift(emb)  # B x C
        B, C = latents.shape[:2]
        if latents.dim() == 4:
            _, _, H, W = latents.shape
            scale = scale.view(B, C, 1, 1)
            shift = shift.view(B, C, 1, 1)
        else:
            scale = scale.view(B, C)
            shift = shift.view(B, C)

        # apply FiLM modulation: (1 + scale) * latents + shift
        out = latents * (1.0 + scale) + shift
        return out
