# pylint: disable=line-too-long, missing-docstring, import-error, import-outside-toplevel, invalid-name

import logging
import os
import random
from dataclasses import dataclass, field
from itertools import chain
from typing import Any

import numpy as np
import torch
import torch.distributed as dist
import torchaudio
from coqpit import Coqpit
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.sampler import WeightedRandomSampler
from trainer.torch import DistributedSampler, DistributedSamplerWrapper
from trainer.trainer_utils import get_optimizer, get_scheduler

from TTS.tts.datasets.dataset import TTSDataset, get_attribute_balancer_weights
from TTS.tts.layers.vits.discriminator import VitsDiscriminator
from TTS.tts.layers.vits.networks import PosteriorEncoder, PriorEncoder, ResidualCouplingBlocks
from TTS.tts.models.base_tts import BaseTTS
from TTS.tts.utils.helpers import rand_segments, segment, sequence_mask
from TTS.utils.audio import AudioProcessor
from TTS.utils.audio.torch_transforms import spec_to_mel, wav_to_mel, wav_to_spec
from TTS.utils.samplers import BucketBatchSampler
from TTS.vocoder.models.hifigan_generator import HifiganGenerator
from TTS.vocoder.utils.generic_utils import plot_results

logger = logging.getLogger(__name__)

##############################
# IO / Feature extraction
##############################


def load_audio(file_path):
    """Load the audio file normalized in [-1, 1]

    Return Shapes:
        - x: :math:`[1, T]`
    """
    x, sr = torchaudio.load(file_path)
    assert (x > 1).sum() + (x < -1).sum() == 0
    return x, sr


#############################
# CONFIGS
#############################


@dataclass
class NativeTTSAudioConfig(Coqpit):
    """Audio config for Native TTS (16kHz, 20ms frames, linear spectrograms)"""
    fft_size: int = 1024
    sample_rate: int = 16000
    win_length: int = 1024
    hop_length: int = 320  # 20ms at 16kHz
    num_mels: int = 80
    mel_fmin: int = 0
    mel_fmax: int | None = None


##############################
# DATASET
##############################


class NativeTTSDataset(Dataset):
    """Dataset for Native TTS that loads preprocessed MFA, F0, and speaker embeddings.

    This is a standalone dataset that doesn't inherit from TTSDataset since Native TTS
    has fundamentally different requirements (no tokenizer, pre-aligned data, etc.).
    """

    def __init__(
        self,
        model_args,
        samples,
        batch_group_size=0,
        min_text_len=0,
        max_text_len=float("inf"),
        min_audio_len=0,
        max_audio_len=float("inf"),
        phoneme_cache_path=None,  # Not used but kept for compatibility
        precompute_num_workers=0,  # Not used but kept for compatibility
        tokenizer=None,  # Not used but kept for compatibility
        start_by_longest=False,
    ):
        super().__init__()

        self.model_args = model_args
        self._samples = samples
        self.batch_group_size = batch_group_size
        # Rename for clarity - these actually filter phoneme sequences in Native TTS
        self.min_phoneme_len = min_text_len  # min_text_len actually means min phoneme length
        self.max_phoneme_len = max_text_len  # max_text_len actually means max phoneme length
        self.min_audio_len = min_audio_len
        self.max_audio_len = max_audio_len
        self.start_by_longest = start_by_longest

    @property
    def samples(self):
        return self._samples

    @samples.setter
    def samples(self, new_samples):
        self._samples = new_samples

    def __len__(self):
        return len(self.samples)

    def preprocess_samples(self):
        """Preprocessing for Native TTS dataset - filter and sort samples."""
        logger.info("Preprocessing Native TTS samples...")

        def get_phoneme_length(mfa_file):
            """Get phoneme sequence length from MFA file."""
            try:
                if os.path.exists(mfa_file):
                    phonemes = np.load(mfa_file)
                    return len(phonemes)
                else:
                    return 0
            except Exception:
                return 0

        # Compute lengths for all samples
        # Note: We use phoneme length as a proxy for audio length since they're aligned
        # This is MUCH faster than scanning 65k audio files with torchaudio.info()
        new_samples = []
        logger.info("Loading phoneme lengths from %d samples...", len(self.samples))

        for idx, item in enumerate(self.samples):
            if idx % 10000 == 0 and idx > 0:
                logger.info("  Processed %d/%d samples...", idx, len(self.samples))

            # Get phoneme sequence length from MFA file
            mfa_file = item.get("mfa_file")
            if not mfa_file:
                continue

            phoneme_length = get_phoneme_length(mfa_file)
            if phoneme_length == 0:
                continue

            # Use phoneme_length * hop_length as approximate audio length
            # Each phoneme frame = 20ms = 320 samples at 16kHz
            item["phoneme_length"] = phoneme_length
            item["audio_length"] = phoneme_length * 320  # Approximate audio samples
            new_samples.append(item)

        logger.info("Loaded %d valid samples out of %d", len(new_samples), len(self.samples))
        samples = new_samples

        # Filter by length
        phoneme_lengths = [i["phoneme_length"] for i in samples]
        audio_lengths = [i["audio_length"] for i in samples]

        # Collect indices to keep
        keep_idx = []
        for idx, (phn_len, aud_len) in enumerate(zip(phoneme_lengths, audio_lengths)):
            if (self.min_phoneme_len <= phn_len <= self.max_phoneme_len and
                self.min_audio_len <= aud_len <= self.max_audio_len):
                keep_idx.append(idx)

        samples = [samples[idx] for idx in keep_idx]

        if len(samples) == 0:
            raise RuntimeError("No samples left after filtering.")

        # Sort by audio length
        samples = sorted(samples, key=lambda x: x["audio_length"])

        if self.start_by_longest:
            # Move longest to beginning
            samples = [samples[-1]] + samples[:-1]

        # Create buckets for batch grouping
        if self.batch_group_size > 0:
            for i in range(len(samples) // self.batch_group_size):
                offset = i * self.batch_group_size
                end_offset = offset + self.batch_group_size
                temp_items = samples[offset:end_offset]
                random.shuffle(temp_items)
                samples[offset:end_offset] = temp_items

        # Update samples
        self.samples = samples

        # Log statistics
        audio_lengths = [s["audio_length"] for s in samples]
        phoneme_lengths = [s["phoneme_length"] for s in samples]

        logger.info("Preprocessing Native TTS samples")
        logger.info("Total samples after filtering: %d", len(samples))
        logger.info("Max audio length: %.2f", np.max(audio_lengths))
        logger.info("Min audio length: %.2f", np.min(audio_lengths))
        logger.info("Avg audio length: %.2f", np.mean(audio_lengths))
        logger.info("Max phoneme length: %d", np.max(phoneme_lengths))
        logger.info("Min phoneme length: %d", np.min(phoneme_lengths))
        logger.info("Avg phoneme length: %.2f", np.mean(phoneme_lengths))
        logger.info("Batch group size: %d", self.batch_group_size)

    def __getitem__(self, idx):
        """Get a single sample - Native TTS requires MFA, F0, and ECAPA embeddings."""
        if self.samples is None:
            raise RuntimeError("Dataset samples not initialized")

        # Handle index out of bounds
        if idx >= len(self.samples):
            idx = idx % len(self.samples)

        item = self.samples[idx]
        wav_filename = os.path.basename(item["audio_file"])

        try:
            # Load audio
            wav, _ = load_audio(item["audio_file"])

            # Load MFA-aligned phonemes (required)
            mfa_path = item.get("mfa_file", None)
            if not mfa_path or not os.path.exists(mfa_path):
                raise FileNotFoundError(
                    f"MFA alignment not found: {mfa_path}\n"
                    f"Native TTS requires MFA-aligned phonemes at 20ms frames."
                )

            token_ids = np.load(mfa_path).tolist()

            # Load F0 (required, must match phoneme length)
            f0_path = item.get("f0_file", None)
            if not f0_path or not os.path.exists(f0_path):
                raise FileNotFoundError(
                    f"F0 file not found: {f0_path}\n"
                    f"Run f0_20ms_batch.py from ac_playground to extract F0."
                )

            f0 = np.load(f0_path)

            if len(f0) != len(token_ids):
                raise ValueError(
                    f"{wav_filename}: MFA={len(token_ids)} frames, F0={len(f0)} frames.\n"
                    f"Run fix_phones_lengths.py to align them."
                )

            # Load ECAPA speaker embedding (required, 192-dim)
            spk_emb_path = item.get("speaker_emb_file", None)
            if not spk_emb_path or not os.path.exists(spk_emb_path):
                raise FileNotFoundError(
                    f"Speaker embedding not found: {spk_emb_path}\n"
                    f"Native TTS requires 192-dim ECAPA-TDNN embeddings."
                )

            speaker_emb = np.load(spk_emb_path)
            if speaker_emb.shape[0] != 192:
                raise ValueError(
                    f"{wav_filename}: Expected 192-dim ECAPA, got {speaker_emb.shape[0]}"
                )

            return {
                "token_ids": token_ids,
                "token_len": len(token_ids),
                "wav": wav,
                "wav_file": wav_filename,
                "speaker_name": item["speaker_name"],
                "audio_unique_name": item["audio_unique_name"],
                "f0": f0,
                "speaker_emb": speaker_emb,
            }

        except Exception as e:
            # If there's an error loading this sample, try the next one
            logger.warning("Error loading sample %s: %s. Trying next sample.", wav_filename, str(e))
            return self.__getitem__((idx + 1) % len(self.samples))

    def collate_fn(self, batch):
        """Collate for Native TTS - minimal padding (MFA+F0 pre-aligned)."""
        B = len(batch)
        batch = {k: [dic[k] for dic in batch] for k in batch[0]}

        # Get max lengths
        max_phone_len = max([len(x) for x in batch["token_ids"]])
        max_wav_len = max([w.shape[1] for w in batch["wav"]])

        # Token/phoneme lengths
        token_lens = torch.LongTensor(batch["token_len"])
        token_rel_lens = token_lens / token_lens.max()

        # Waveform lengths
        wav_lens = torch.LongTensor([w.shape[1] for w in batch["wav"]])
        wav_rel_lens = wav_lens / wav_lens.max()

        # Initialize tensors - zero padding
        token_padded = torch.zeros(B, max_phone_len, dtype=torch.long)
        wav_padded = torch.zeros(B, 1, max_wav_len)
        f0_padded = torch.zeros(B, max_phone_len)

        # Fixed-size speaker embeddings - no padding needed
        speaker_emb_batch = torch.stack([
            torch.FloatTensor(batch["speaker_emb"][i]) for i in range(B)
        ])

        # Fill tensors
        for i in range(B):
            phone_len = batch["token_len"][i]

            # Phonemes
            token_padded[i, :phone_len] = torch.LongTensor(batch["token_ids"][i])

            # Waveform
            wav = batch["wav"][i]
            wav_padded[i, :, :wav.size(1)] = torch.FloatTensor(wav)

            # F0 - must match phoneme length
            f0 = batch["f0"][i]
            if len(f0) != phone_len:
                raise RuntimeError(
                    f"F0 length mismatch in batch item {i} ({batch['audio_unique_name'][i]}): "
                    f"F0={len(f0)}, phonemes={phone_len}. Data corruption?"
                )
            f0_padded[i, :phone_len] = torch.FloatTensor(f0)

        return {
            "tokens": token_padded,
            "token_lens": token_lens,
            "token_rel_lens": token_rel_lens,
            "waveform": wav_padded,
            "waveform_lens": wav_lens,
            "waveform_rel_lens": wav_rel_lens,
            "f0": f0_padded,
            "speaker_emb": speaker_emb_batch,
            "speaker_names": batch["speaker_name"],
            "audio_files": batch["wav_file"],
            "audio_unique_names": batch["audio_unique_name"],
        }


##############################
# MODEL DEFINITION
##############################


@dataclass
class NativeTTSArgs(Coqpit):
    """Native TTS model arguments.

    Based on https://arxiv.org/abs/2506.16580

    Native TTS uses:
    - PriorEncoder with F0 conditioning (replaces TextEncoder + Duration Predictor)
    - MFA pre-aligned phonemes at 20ms frames (no MAS needed)
    - ECAPA speaker embeddings (192-dim, required, pre-computed)
    - Linear spectrograms (513 channels) for posterior encoder
    - 16kHz audio with 320 hop length (20ms frames)
    """

    # Phoneme vocabulary
    num_chars: int = 100  # overridden at runtime from MFA phoneme_map.json

    # Architecture
    out_channels: int = 513  # Linear spectrogram (FFT/2 + 1) for 1024 FFT
    hidden_channels: int = 192
    spec_segment_size: int = 32

    # Prior Encoder (phonemes + F0 → prior distribution)
    hidden_channels_ffn_prior: int = 768
    num_heads_prior: int = 2
    num_layers_prior: int = 6
    kernel_size_prior: int = 3
    dropout_p_prior: float = 0.1
    f0_embedding_dim: int = 2  # F0 embedding (changed from 1 to make 194 channels divisible by 2 heads)

    # Posterior Encoder (linear spec + speaker → posterior distribution)
    kernel_size_posterior_encoder: int = 5
    dilation_rate_posterior_encoder: int = 1
    num_layers_posterior_encoder: int = 16

    # Flow (maps posterior → prior space)
    kernel_size_flow: int = 5
    dilation_rate_flow: int = 1
    num_layers_flow: int = 4

    # HiFi-GAN Decoder (latent → waveform)
    resblock_type_decoder: str = "1"
    resblock_kernel_sizes_decoder: list[int] = field(default_factory=lambda: [3, 7, 11])
    resblock_dilation_sizes_decoder: list[list[int]] = field(
        default_factory=lambda: [[1, 3, 5], [1, 3, 5], [1, 3, 5]]
    )
    upsample_rates_decoder: list[int] = field(default_factory=lambda: [8, 8, 5, 1])  # Product = 320
    upsample_initial_channel_decoder: int = 512
    upsample_kernel_sizes_decoder: list[int] = field(default_factory=lambda: [16, 16, 10, 2])

    # Discriminator
    periods_multi_period_discriminator: list[int] = field(default_factory=lambda: [2, 3, 5, 7, 11])
    use_spectral_norm_disriminator: bool = False
    init_discriminator: bool = True

    # Training
    noise_scale: float = 1.0
    inference_noise_scale: float = 0.667
    max_inference_len: int | None = None

    # Multi-speaker (ECAPA embeddings required, pre-computed)
    embedded_speaker_dim: int = 192  # ECAPA-TDNN output dimension (fixed)


class NativeTTS(BaseTTS):
    """Native TTS Model - Simplified VITS for accent correction.

    Based on https://arxiv.org/abs/2506.16580

    This model uses:
    - MFA phoneme alignments instead of text
    - F0 (pitch) features
    - Speaker embeddings from a pre-trained encoder
    """

    def __init__(self, config: Coqpit, ap: AudioProcessor | None = None):
        super().__init__(config, ap, tokenizer=None)

        # Native TTS uses pre-computed ECAPA embeddings (192-dim)
        self.embedded_speaker_dim = self.args.embedded_speaker_dim

        self.noise_scale = self.args.noise_scale
        self.inference_noise_scale = self.args.inference_noise_scale
        self.max_inference_len = self.args.max_inference_len
        self.spec_segment_size = self.args.spec_segment_size

        # Prior Encoder (replaces TextEncoder + Duration Predictor)
        self.prior_encoder = PriorEncoder(
            self.args.num_chars,
            self.args.hidden_channels,
            self.args.hidden_channels,
            self.args.hidden_channels_ffn_prior,
            self.args.num_heads_prior,
            self.args.num_layers_prior,
            self.args.kernel_size_prior,
            self.args.dropout_p_prior,
            f0_embedding_dim=self.args.f0_embedding_dim,
        )

        # Posterior Encoder (conditioned on speaker)
        self.posterior_encoder = PosteriorEncoder(
            self.args.out_channels,  # Linear spectrogram channels (513)
            self.args.hidden_channels,
            self.args.hidden_channels,
            kernel_size=self.args.kernel_size_posterior_encoder,
            dilation_rate=self.args.dilation_rate_posterior_encoder,
            num_layers=self.args.num_layers_posterior_encoder,
            cond_channels=self.embedded_speaker_dim,
        )

        # Flow (conditioned on speaker)
        self.flow = ResidualCouplingBlocks(
            self.args.hidden_channels,
            self.args.hidden_channels,
            kernel_size=self.args.kernel_size_flow,
            dilation_rate=self.args.dilation_rate_flow,
            num_layers=self.args.num_layers_flow,
            cond_channels=self.embedded_speaker_dim,
        )

        # HiFi-GAN Decoder (conditioned on speaker)
        self.waveform_decoder = HifiganGenerator(
            self.args.hidden_channels,
            1,
            self.args.resblock_type_decoder,
            self.args.resblock_dilation_sizes_decoder,
            self.args.resblock_kernel_sizes_decoder,
            self.args.upsample_kernel_sizes_decoder,
            self.args.upsample_initial_channel_decoder,
            self.args.upsample_rates_decoder,
            inference_padding=0,
            cond_channels=self.embedded_speaker_dim,
            conv_pre_weight_norm=False,
            conv_post_weight_norm=False,
            conv_post_bias=False,
        )

        # Discriminator
        if self.args.init_discriminator:
            self.disc = VitsDiscriminator(
                periods=self.args.periods_multi_period_discriminator,
                use_spectral_norm=self.args.use_spectral_norm_disriminator,
            )

    def forward(  # pylint: disable=dangerous-default-value
        self,
        x: torch.Tensor,
        x_lengths: torch.Tensor,
        y: torch.Tensor,
        y_lengths: torch.Tensor,
        waveform: torch.Tensor,
        aux_input: dict[str, Any]
    ) -> dict:
        """Native TTS forward pass - no MAS, no duration prediction."""

        if aux_input is None:
            aux_input = {}

        # Get F0 and speaker embedding (required)
        f0 = aux_input.get("f0")
        speaker_emb = aux_input.get("speaker_emb")

        if f0 is None:
            raise ValueError("Native TTS requires 'f0' in aux_input")
        if speaker_emb is None:
            raise ValueError("Native TTS requires 'speaker_emb' in aux_input")

        # Prepare speaker embedding: [B, 192] → [B, 192, 1]
        g = speaker_emb.unsqueeze(-1) if speaker_emb.dim() == 2 else speaker_emb

        # Prepare F0: [B, T] → [B, 1, T] for conv1d in PriorEncoder
        if f0.dim() == 2:
            f0 = f0.unsqueeze(1)  # [B, T] → [B, 1, T]

        # Prior encoder: MFA phonemes + F0 → prior distribution
        # Returns (x, m, logs, x_mask) like VITS TextEncoder
        # We ignore x and x_mask since Native TTS doesn't use MAS/duration predictor
        _, m_p, logs_p, _ = self.prior_encoder(x, x_lengths, f0=f0)

        # Posterior encoder: linear spec + speaker → posterior distribution
        z, m_q, logs_q, y_mask = self.posterior_encoder(y, y_lengths, g=g)

        # Flow: posterior → prior space
        z_p = self.flow(z, y_mask, g=g)

        # Random segment for decoder training
        z_slice, slice_ids = rand_segments(
            z, y_lengths, self.spec_segment_size, let_short_samples=True, pad_short=True
        )

        # Decode to waveform
        o = self.waveform_decoder(z_slice, g=g)

        # Ground truth waveform segment
        wav_seg = segment(
            waveform,
            slice_ids * self.config.audio.hop_length,
            self.spec_segment_size * self.config.audio.hop_length,
            pad_short=True,
        )

        return {
            "model_outputs": o,
            "waveform_seg": wav_seg,
            "z": z,
            "z_p": z_p,
            "m_p": m_p,
            "logs_p": logs_p,
            "m_q": m_q,
            "logs_q": logs_q,
            "slice_ids": slice_ids,
            "loss_duration": torch.tensor(0.0).to(x.device),  # No duration loss
        }

    @staticmethod
    def _set_x_lengths(x, aux_input):
        if aux_input and "x_lengths" in aux_input and aux_input["x_lengths"] is not None:
            return aux_input["x_lengths"]
        return torch.tensor(x.shape[1:2]).to(x.device)

    @torch.inference_mode()
    def inference(
        self,
        input,  # pylint: disable=redefined-builtin
        aux_input=None,
    ):  # pylint: disable=dangerous-default-value
        """Native TTS inference."""
        if aux_input is None:
            aux_input = {}

        # Get F0 and speaker embedding from aux_input
        f0 = aux_input.get("f0")
        if f0 is None:
            raise ValueError("Native TTS inference requires 'f0' in aux_input")

        speaker_emb = aux_input.get("speaker_emb")
        if speaker_emb is None:
            raise ValueError("Native TTS inference requires 'speaker_emb' in aux_input")

        x_lengths = self._set_x_lengths(input, aux_input)

        # Prepare speaker embedding: [B, 192] → [B, 192, 1]
        g = speaker_emb.unsqueeze(-1) if speaker_emb.dim() == 2 else speaker_emb

        # Prepare F0: [B, T] → [B, 1, T]
        if f0.dim() == 2:
            f0 = f0.unsqueeze(1)

        # Prior encoder with MFA phonemes and F0
        # Returns (x, m, logs, x_mask) like VITS TextEncoder
        # We need x_mask for inference but ignore x (no MAS/duration predictor)
        _, m_p, logs_p, x_mask = self.prior_encoder(input, x_lengths, f0=f0)

        # Sample from prior
        z_p = m_p + torch.randn_like(m_p) * torch.exp(logs_p) * self.inference_noise_scale

        # Inverse flow to get z
        y_mask = x_mask  # In Native TTS, alignment is 1:1
        z = self.flow(z_p, y_mask, g=g, reverse=True)

        # Generate waveform
        o = self.waveform_decoder((z * y_mask)[:, :, : self.max_inference_len], g=g)

        return {
            "model_outputs": o,
            "z": z,
            "z_p": z_p,
            "m_p": m_p,
            "logs_p": logs_p,
            "y_mask": y_mask,
        }

    def train_step(self, batch: dict, criterion: nn.Module, optimizer_idx: int) -> tuple[dict, dict]:
        """Perform a single training step."""

        spec_lens = batch["spec_lens"]

        if optimizer_idx == 0:
            tokens = batch["tokens"]
            token_lenghts = batch["token_lens"]
            spec = batch["spec"]
            waveform = batch["waveform"]

            # Prepare aux_input with Native TTS specific inputs
            aux_input = {
                "f0": batch.get("f0"),
                "speaker_emb": batch.get("speaker_emb"),
            }

            # generator pass
            outputs = self.forward(
                tokens,
                token_lenghts,
                spec,
                spec_lens,
                waveform,
                aux_input=aux_input,
            )

            # cache tensors for the generator pass
            self.model_outputs_cache = outputs  # pylint: disable=attribute-defined-outside-init

            # compute scores and features
            scores_disc_fake, _, scores_disc_real, _ = self.disc(
                outputs["model_outputs"].detach(), outputs["waveform_seg"]
            )

            # compute loss
            with torch.autocast("cuda", enabled=False):  # use float32 for the criterion
                loss_dict = criterion[optimizer_idx](
                    scores_disc_real,
                    scores_disc_fake,
                )
            return outputs, loss_dict

        if optimizer_idx == 1:
            mel = batch["mel"]

            # compute melspec segment (outside autocast to avoid inference tensor issues)
            mel_slice = segment(
                mel.float(), self.model_outputs_cache["slice_ids"], self.spec_segment_size, pad_short=True
            )
            mel_slice_hat = wav_to_mel(
                y=self.model_outputs_cache["model_outputs"].float(),
                n_fft=self.config.audio.fft_size,
                sample_rate=self.config.audio.sample_rate,
                num_mels=self.config.audio.num_mels,
                hop_length=self.config.audio.hop_length,
                win_length=self.config.audio.win_length,
                fmin=self.config.audio.mel_fmin,
                fmax=self.config.audio.mel_fmax,
                center=False,
            )

            # compute discriminator scores and features
            scores_disc_fake, feats_disc_fake, _, feats_disc_real = self.disc(
                self.model_outputs_cache["model_outputs"], self.model_outputs_cache["waveform_seg"]
            )

            # compute losses
            with torch.autocast("cuda", enabled=False):  # use float32 for the criterion
                loss_dict = criterion[optimizer_idx](
                    mel_slice_hat=mel_slice_hat.float(),
                    mel_slice=mel_slice.float(),
                    z_p=self.model_outputs_cache["z_p"].float(),
                    logs_q=self.model_outputs_cache["logs_q"].float(),
                    m_p=self.model_outputs_cache["m_p"].float(),
                    logs_p=self.model_outputs_cache["logs_p"].float(),
                    z_len=spec_lens,
                    scores_disc_fake=scores_disc_fake,
                    feats_disc_fake=feats_disc_fake,
                    feats_disc_real=feats_disc_real,
                    loss_duration=self.model_outputs_cache["loss_duration"],
                )

            return self.model_outputs_cache, loss_dict

        raise ValueError(" [!] Unexpected `optimizer_idx`.")

    def format_batch(self, batch: dict) -> dict:
        """Format batch for Native TTS - just pass through, no transformation needed.

        Native TTS uses a custom batch format from NativeTTSDataset.collate_fn,
        so we don't need the base class formatting.
        """
        return batch

    @torch.inference_mode()
    def format_batch_on_device(self, batch):
        """Compute LINEAR spectrograms on device (Native TTS uses x_lin, not mel)."""
        ac = self.config.audio

        wav = batch["waveform"]

        # Compute LINEAR spectrogram (not mel) - used for posterior encoder
        batch["spec"] = wav_to_spec(wav, ac.fft_size, ac.hop_length, ac.win_length, center=False)

        # Compute spectrogram frame lengths
        batch["spec_lens"] = (batch["spec"].shape[2] * batch["waveform_rel_lens"]).int()

        # Zero the padding frames
        batch["spec"] = batch["spec"] * sequence_mask(batch["spec_lens"]).unsqueeze(1)

        # Compute mel for loss computation (standard VITS loss uses mel)
        batch["mel"] = spec_to_mel(
            spec=batch["spec"],
            n_fft=ac.fft_size,
            num_mels=ac.num_mels,
            sample_rate=ac.sample_rate,
            fmin=ac.mel_fmin,
            fmax=ac.mel_fmax,
        )
        batch["mel_lens"] = batch["spec_lens"]  # Same length
        batch["mel"] = batch["mel"] * sequence_mask(batch["mel_lens"]).unsqueeze(1)

        return batch

    def get_sampler(self, config: Coqpit, dataset: TTSDataset, num_gpus=1, is_eval=False):
        weights = None
        data_items = dataset.samples

        if data_items is None:
            raise RuntimeError("Dataset samples not initialized")

        if getattr(config, "use_weighted_sampler", False):
            for attr_name, alpha in config.weighted_sampler_attrs.items():
                logger.info("Using weighted sampler for '%s' with alpha %.3f", attr_name, alpha)
                multi_dict = config.weighted_sampler_multipliers.get(attr_name, None)
                weights, attr_names, attr_weights = get_attribute_balancer_weights(
                    attr_name=attr_name, items=data_items, multi_dict=multi_dict
                )
                weights = weights * alpha
                logger.info("Weights for '%s': %s", attr_names, attr_weights)

        if weights is not None:
            w_sampler = WeightedRandomSampler(weights, len(weights))
            batch_sampler = BucketBatchSampler(
                w_sampler,
                data=data_items,
                batch_size=config.eval_batch_size if is_eval else config.batch_size,
                sort_key=lambda x: os.path.getsize(x["audio_file"]),
                drop_last=True,
            )
        else:
            batch_sampler = None

        if batch_sampler is None:
            return DistributedSampler(dataset) if num_gpus > 1 else None

        return DistributedSamplerWrapper(batch_sampler) if num_gpus > 1 else batch_sampler

    def get_data_loader(
        self,
        config: Coqpit,
        assets: dict,  # pylint: disable=unused-argument
        is_eval: bool,
        samples: list[dict] | list[list],
        verbose: bool,  # pylint: disable=unused-argument
        num_gpus: int,
        rank: int | None = None,  # pylint: disable=unused-argument
    ) -> "DataLoader":
        """Initialize data loader for Native TTS."""
        if is_eval and not config.run_eval:
            return None

        dataset = NativeTTSDataset(
            model_args=self.args,
            samples=samples,
            batch_group_size=0 if is_eval else config.batch_group_size * config.batch_size,
            min_text_len=config.min_text_len,
            max_text_len=config.max_text_len,
            min_audio_len=config.min_audio_len,
            max_audio_len=config.max_audio_len,
            phoneme_cache_path=config.phoneme_cache_path,
            precompute_num_workers=config.precompute_num_workers,
            tokenizer=None,  # No tokenizer in Native TTS
            start_by_longest=config.start_by_longest,
        )

        if num_gpus > 1:
            dist.barrier()

        dataset.preprocess_samples()
        sampler = self.get_sampler(config, dataset, num_gpus)

        if sampler is None:
            return DataLoader(
                dataset,
                batch_size=config.eval_batch_size if is_eval else config.batch_size,
                shuffle=False,
                collate_fn=dataset.collate_fn,
                drop_last=False,
                num_workers=config.num_eval_loader_workers if is_eval else config.num_loader_workers,
                pin_memory=False,
            )

        if num_gpus > 1:
            return DataLoader(
                dataset,
                sampler=sampler,
                batch_size=config.eval_batch_size if is_eval else config.batch_size,
                collate_fn=dataset.collate_fn,
                num_workers=config.num_eval_loader_workers if is_eval else config.num_loader_workers,
                pin_memory=False,
            )

        return DataLoader(
            dataset,
            batch_sampler=sampler,
            collate_fn=dataset.collate_fn,
            num_workers=config.num_eval_loader_workers if is_eval else config.num_loader_workers,
            pin_memory=False,
        )

    def get_optimizer(self) -> list:
        """Initiate and return the GAN optimizers."""
        optimizer0 = get_optimizer(self.config.optimizer, self.config.optimizer_params, self.config.lr_disc, self.disc)

        gen_parameters = chain(params for k, params in self.named_parameters() if not k.startswith("disc."))
        optimizer1 = get_optimizer(
            self.config.optimizer, self.config.optimizer_params, self.config.lr_gen, parameters=gen_parameters
        )
        return [optimizer0, optimizer1]

    def get_lr(self) -> list:
        """Set the initial learning rates for each optimizer."""
        return [self.config.lr_disc, self.config.lr_gen]

    def get_scheduler(self, optimizer) -> list:
        """Set the schedulers for each optimizer."""
        scheduler_D = get_scheduler(self.config.lr_scheduler_disc, self.config.lr_scheduler_disc_params, optimizer[0])
        scheduler_G = get_scheduler(self.config.lr_scheduler_gen, self.config.lr_scheduler_gen_params, optimizer[1])
        return [scheduler_D, scheduler_G]

    def _create_logs(self, batch, outputs: dict[str, Any] | list[dict[str, Any]]):
        """Create training logs for TensorBoard.

        Args:
            batch: Training batch data
            outputs: Outputs from discriminator and generator steps (list or dict)

        Returns:
            Tuple of (figures_dict, audios_dict)
        """
        # Handle both list and dict formats
        if isinstance(outputs, list):
            outputs_dict = outputs[1]  # Get generator step
        else:
            outputs_dict = outputs

        # Get waveforms from generator step
        y_hat = outputs_dict["model_outputs"]
        y = outputs_dict["waveform_seg"]

        # Plot waveform comparison
        figures = plot_results(y_hat, y, self.ap)

        # Sample audio for logging
        sample_voice = y_hat[0].squeeze(0).detach().cpu().numpy()
        audios = {"audio": sample_voice}

        return figures, audios

    @torch.inference_mode()
    def test_run(self, assets) -> dict[str, Any]:
        """Native TTS test run - no test sentences needed.

        Native TTS uses preprocessed data, so we skip test synthesis.

        Returns:
            Empty dictionary for figures and audios
        """
        logger.info("Native TTS: Skipping test run (uses preprocessed data)")
        return {"figures": {}, "audios": {}}

    @torch.inference_mode()
    def eval_step(self, batch: dict, criterion: nn.Module, optimizer_idx: int | None = None):
        """Evaluation step - similar to train_step but without gradients.

        Args:
            batch: Evaluation batch
            criterion: Loss criterion
            optimizer_idx: Index of optimizer (0=discriminator, 1=generator)

        Returns:
            Tuple of (outputs, loss_dict)
        """
        if optimizer_idx is None:
            optimizer_idx = 0
        return self.train_step(batch, criterion, optimizer_idx)

    def get_criterion(self):
        """Get criterions for each optimizer."""
        from TTS.tts.layers.losses import (  # pylint: disable=import-outside-toplevel
            VitsDiscriminatorLoss,
            VitsGeneratorLoss,
        )

        return [VitsDiscriminatorLoss(self.config), VitsGeneratorLoss(self.config)]

    @staticmethod
    def init_from_config(config: NativeTTSAudioConfig):
        """Initiate model from config."""
        upsample_rate = torch.prod(torch.as_tensor(config.model_args.upsample_rates_decoder)).item()
        assert upsample_rate == config.audio.hop_length, (
            f"Product of upsample rates must equal hop length: {upsample_rate} vs {config.audio.hop_length}"
        )

        ap = AudioProcessor.init_from_config(config)
        return NativeTTS(config, ap)
