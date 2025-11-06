# pylint: disable=line-too-long, missing-module-docstring
# python recipes/ljspeech/vits_tts/train_native_tts.py --data_path /path/to/augmented_dataset
import argparse
import glob
import json
import os
from pathlib import Path

import torch
from trainer import Trainer, TrainerArgs

from TTS.tts.configs.shared_configs import BaseDatasetConfig
from TTS.tts.configs.native_tts_config import NativeTTSConfig
from TTS.tts.models.native_tts import NativeTTS, NativeTTSArgs, NativeTTSAudioConfig  # Changed imports
from TTS.tts.datasets import load_tts_samples
from TTS.utils.audio import AudioProcessor

output_path = os.path.dirname(os.path.abspath(__file__))


def get_hardware_config(vram_gb=None, vcpus=None):
    """Calculate optimal batch size and workers based on hardware specs."""
    num_gpus = torch.cuda.device_count() if torch.cuda.is_available() else 0
    if vram_gb is None and num_gpus > 0:
        vram_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)
    if vcpus is None:
        vcpus = os.cpu_count() or 8
    batch_size = 2 * max(8, int((vram_gb - 2) * 1.1)) if vram_gb else 32
    batch_size = min(batch_size, 32)
    num_workers = max(4, min(vcpus - 2, 8))
    return {
        "num_gpus": num_gpus,
        "batch_size": batch_size,
        "eval_batch_size": max(8, batch_size // 2),
        "num_workers": num_workers,
        "eval_workers": max(2, num_workers // 3),
    }


def calculate_num_chars(mfa_dir):
    """Calculate num_chars from MFA phoneme_map.json."""
    phoneme_map_path = Path(mfa_dir) / "phoneme_map.json"

    if not phoneme_map_path.exists():
        raise FileNotFoundError(
            f"❌ FATAL: phoneme_map.json not found at {phoneme_map_path}\n"
            f"   This file is required and should be created by mfa_upsample_batch.py\n"
            f"   Please run MFA alignment preprocessing first."
        )

    with phoneme_map_path.open("r", encoding="utf-8") as f:
        phoneme_map = json.load(f)

    if not phoneme_map:
        raise ValueError(
            f"❌ FATAL: phoneme_map.json at {phoneme_map_path} is empty!\n"
            f"   Cannot determine vocabulary size. Check MFA preprocessing."
        )

    max_phoneme_id = max(phoneme_map.values())
    num_chars = max_phoneme_id + 1

    print(f"✅ Loaded phoneme_map.json: {len(phoneme_map)} phonemes, num_chars={num_chars}")
    print(f"   (max phoneme ID: {max_phoneme_id}, 'sil' → {phoneme_map.get('sil', 'N/A')})")

    return num_chars


def native_tts_formatter(root_path, meta_file=None, **kwargs):  # pylint: disable=unused-argument
    """Custom formatter for Native TTS augmented dataset."""
    items = []

    wav_dir = os.path.join(root_path, "wavs_16k")
    mfa_dir = os.path.join(root_path, "mfa_alignments")
    f0_dir = os.path.join(root_path, "f0_features")
    spk_emb_dir = os.path.join(root_path, "speaker_embeddings")

    wav_files = glob.glob(os.path.join(wav_dir, "*.wav"))

    for wav_file in wav_files:
        basename = Path(wav_file).stem
        speaker_id = basename.split("_")[-1]

        mfa_file = os.path.join(mfa_dir, f"{basename}.npy")
        f0_file = os.path.join(f0_dir, f"{basename}.npy")
        spk_emb_file = os.path.join(spk_emb_dir, f"{speaker_id}.npy")

        if not os.path.exists(mfa_file):
            print(f"Warning: MFA file not found: {mfa_file}")
            continue
        if not os.path.exists(f0_file):
            print(f"Warning: F0 file not found: {f0_file}")
            continue
        if not os.path.exists(spk_emb_file):
            print(f"Warning: Speaker embedding not found: {spk_emb_file}")
            continue

        items.append(
            {
                "audio_file": wav_file,
                "mfa_file": mfa_file,
                "f0_file": f0_file,
                "speaker_emb_file": spk_emb_file,
                "speaker_name": speaker_id,
                "text": "",  # No text needed
                "audio_unique_name": basename,
            }
        )

    return items


def main():
    """Train Native TTS model on augmented LJSpeech dataset."""
    parser = argparse.ArgumentParser(description="Train Native TTS model")
    parser.add_argument(
        "--data_path",
        type=str,
        default="/workspace/augmented_dataset",
        help="Path to augmented dataset root directory",
    )
    parser.add_argument("--vram", type=float, default=None, help="GPU VRAM in GB")
    parser.add_argument("--vcpus", type=int, default=None, help="Number of CPU cores")
    args = parser.parse_args()

    hw_config = get_hardware_config(vram_gb=args.vram, vcpus=args.vcpus)
    print(f"Hardware config: {hw_config}")

    mfa_dir = os.path.join(args.data_path, "mfa_alignments")
    num_chars = calculate_num_chars(mfa_dir)

    dataset_config = BaseDatasetConfig(
        formatter="native_tts_augmented",
        meta_file_train=None,
        path=args.data_path,
    )

    audio_config = NativeTTSAudioConfig(
        sample_rate=16000,
        win_length=1024,
        hop_length=320,
        num_mels=80,
        mel_fmin=0,
        mel_fmax=None,
    )

    model_args = NativeTTSArgs(
        num_chars=num_chars,
        embedded_speaker_dim=192,
        upsample_rates_decoder=[8, 8, 5, 1],
        upsample_kernel_sizes_decoder=[16, 16, 10, 2],
    )

    # pylint: disable=unexpected-keyword-arg
    config = NativeTTSConfig(
        model_args=model_args,
        audio=audio_config,
        run_name="native_tts_ljspeech_freevc_vctk",
        batch_size=hw_config["batch_size"],
        eval_batch_size=hw_config["eval_batch_size"],
        batch_group_size=5,
        num_loader_workers=hw_config["num_workers"],
        num_eval_loader_workers=hw_config["eval_workers"],
        run_eval=True,
        test_delay_epochs=-1,
        epochs=1000,
        text_cleaner=None,
        use_phonemes=False,
        phoneme_language=None,
        compute_input_seq_cache=False,
        print_step=25,
        print_eval=True,
        mixed_precision=True,
        output_path=output_path,
        datasets=[dataset_config],
        cudnn_benchmark=False,
        test_sentences=[],
    )
    # pylint: enable=unexpected-keyword-arg

    ap = AudioProcessor.init_from_config(config)

    train_samples, eval_samples = load_tts_samples(
        dataset_config,
        eval_split=True,
        eval_split_max_size=config.eval_split_max_size,
        eval_split_size=config.eval_split_size,
        formatter=native_tts_formatter,
    )

    print(f"Loaded {len(train_samples)} training samples and {len(eval_samples)} eval samples")
    if len(train_samples) > 0:
        print(f"Sample: {train_samples[0]}")

    # Initialize model
    model = NativeTTS(config, ap)

    # Train
    trainer = Trainer(
        TrainerArgs(),
        config,
        output_path,
        model=model,
        train_samples=train_samples,
        eval_samples=eval_samples,
    )
    trainer.fit()


if __name__ == "__main__":
    main()
