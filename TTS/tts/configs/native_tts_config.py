from dataclasses import dataclass, field

from TTS.tts.configs.shared_configs import BaseTTSConfig
from TTS.tts.models.native_tts import NativeTTSArgs, NativeTTSAudioConfig


@dataclass
class NativeTTSConfig(BaseTTSConfig):
    """Defines parameters for Native TTS model."""

    model: str = "native_tts"
    model_args: NativeTTSArgs = field(default_factory=NativeTTSArgs)
    audio: NativeTTSAudioConfig = field(default_factory=NativeTTSAudioConfig)  # type: ignore[assignment]

    # optimizer
    grad_clip: list[float] = field(default_factory=lambda: [1000, 1000])
    lr_gen: float = 0.0002
    lr_disc: float = 0.0002
    lr_scheduler_gen: str = "ExponentialLR"
    lr_scheduler_gen_params: dict = field(default_factory=lambda: {"gamma": 0.999875, "last_epoch": -1})
    lr_scheduler_disc: str = "ExponentialLR"
    lr_scheduler_disc_params: dict = field(default_factory=lambda: {"gamma": 0.999875, "last_epoch": -1})
    scheduler_after_epoch: bool = True
    optimizer: str = "AdamW"
    optimizer_params: dict = field(default_factory=lambda: {"betas": [0.8, 0.99], "eps": 1e-9, "weight_decay": 0.01})

    # loss params
    kl_loss_alpha: float = 1.0
    disc_loss_alpha: float = 1.0
    gen_loss_alpha: float = 1.0
    feat_loss_alpha: float = 1.0
    mel_loss_alpha: float = 45.0

    # data loader params
    return_wav: bool = True
    compute_linear_spec: bool = True  # Required for Native TTS posterior encoder

    # sampler params
    use_weighted_sampler: bool = False
    weighted_sampler_attrs: dict = field(default_factory=lambda: {})
    weighted_sampler_multipliers: dict = field(default_factory=lambda: {})

    # overrides
    r: int = 1

    # testing
    test_sentences: list[str] | list[list[str]] = field(default_factory=list)