# pylint: disable=line-too-long, missing-module-docstring
"""
Native TTS Inference Script

This script uses a trained Native TTS model to generate native-accented audio
from preprocessed inputs (MFA alignments, F0 features, speaker embeddings).

Used for creating ground truth datasets for Accent Conversion training:
- Input: Non-native audio (e.g., ARCTIC) with MFA/F0/speaker embeddings
- Output: Native-accented audio with same speaker identity

Example:
    python infer_native_tts.py \\
        --checkpoint /path/to/checkpoint.pth \\
        --config /path/to/config.json \\
        --mfa-dir /path/to/arctic_mfa_alignments \\
        --f0-dir /path/to/arctic_f0_features \\
        --speaker-emb /path/to/speaker_embedding.npy \\
        --out-dir /path/to/output_wavs \\
        --device cuda
"""

import argparse
import json
import os
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
from tqdm import tqdm

from TTS.tts.configs.native_tts_config import NativeTTSConfig
from TTS.tts.models.native_tts import NativeTTS
from TTS.utils.audio import AudioProcessor


def load_model(checkpoint_path: str, config_path: str, device: str = "cuda"):
    """Load trained Native TTS model from checkpoint.

    Args:
        checkpoint_path: Path to model checkpoint (.pth)
        config_path: Path to config file (.json)
        device: Device to load model on

    Returns:
        model: Loaded NativeTTS model in eval mode
        ap: AudioProcessor for the model
    """
    # Load config
    with open(config_path, "r", encoding="utf-8") as f:
        config_dict = json.load(f)

    config = NativeTTSConfig()
    config.from_dict(config_dict)

    # Initialize audio processor
    ap = AudioProcessor.init_from_config(config)

    # Initialize model
    model = NativeTTS.init_from_config(config)
    model.ap = ap  # Set audio processor

    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    model.load_state_dict(checkpoint["model"])

    model = model.to(device)
    model.eval()

    print(f"✅ Loaded model from {checkpoint_path}")
    print(f"   Trained for {checkpoint.get('step', 'unknown')} steps")

    return model, ap


def infer_batch(
    model: NativeTTS,
    mfa_files: list[str],
    f0_files: list[str],
    speaker_emb: np.ndarray,
    device: str = "cuda",
    inference_noise_scale: float = 0.5,
):
    """Run inference on a batch of files.

    Args:
        model: Trained NativeTTS model
        mfa_files: List of paths to MFA .npy files
        f0_files: List of paths to F0 .npy files
        speaker_emb: Speaker embedding array [192]
        device: Device for inference

    Returns:
        wavs: List of generated waveforms
        basenames: List of file basenames
    """
    wavs = []
    basenames = []

    for mfa_path, f0_path in tqdm(zip(mfa_files, f0_files), total=len(mfa_files), desc="Inference"):
        basename = Path(mfa_path).stem
        basenames.append(basename)

        # Load inputs
        phoneme_ids = np.load(mfa_path)  # [T] int16
        f0_values = np.load(f0_path)      # [T] float32

        # Validate lengths match
        if len(phoneme_ids) != len(f0_values):
            print(f"⚠️  Warning: {basename} - MFA length {len(phoneme_ids)} != F0 length {len(f0_values)}, skipping")
            wavs.append(None)
            continue

        # Prepare tensors
        x = torch.LongTensor(phoneme_ids).unsqueeze(0).to(device)  # [1, T]
        f0 = torch.FloatTensor(f0_values).unsqueeze(0).to(device)  # [1, T]
        spk_emb = torch.FloatTensor(speaker_emb).unsqueeze(0).to(device)  # [1, 192]

        # Prepare aux_input for inference
        aux_input = {
            "f0": f0,
            "speaker_emb": spk_emb,
        }

        # Run inference (no gradient needed)
        with torch.no_grad():
            # Temporarily modify inference noise scale for clearer speech
            original_noise_scale = model.inference_noise_scale
            model.inference_noise_scale = inference_noise_scale  # Use parameter value
            outputs = model.inference(x, aux_input=aux_input)
            model.inference_noise_scale = original_noise_scale  # Restore original

        # Extract waveform
        wav = outputs["model_outputs"].squeeze().cpu().numpy()
        wavs.append(wav)

    return wavs, basenames


def main():
    """Main inference function."""
    parser = argparse.ArgumentParser(description="Native TTS Inference")
    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Path to model checkpoint (.pth)",
    )
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to config file (.json)",
    )
    parser.add_argument(
        "--mfa-dir",
        type=str,
        required=True,
        help="Directory containing MFA alignment .npy files",
    )
    parser.add_argument(
        "--f0-dir",
        type=str,
        required=True,
        help="Directory containing F0 feature .npy files",
    )
    parser.add_argument(
        "--speaker-emb-dir",
        type=str,
        required=True,
        help="Directory containing speaker embedding .npy files (192-dim ECAPA)",
    )
    parser.add_argument(
        "--out-dir",
        type=str,
        required=True,
        help="Output directory for generated WAV files",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device to use for inference (cuda/cpu)",
    )
    parser.add_argument(
        "--inference-noise-scale",
        type=float,
        default=0.5,
        help="Noise scale for inference (lower = clearer speech, default 0.5)",
    )

    args = parser.parse_args()

    # Create output directory
    os.makedirs(args.out_dir, exist_ok=True)

    # Load model
    print("Loading model...")
    model, ap = load_model(args.checkpoint, args.config, args.device)

    # Loop over all speaker embeddings in the directory
    emb_dir = Path(args.speaker_emb_dir)
    emb_files = sorted([f for f in emb_dir.glob('*.npy')])
    if not emb_files:
        print(f"❌ Error: No speaker embedding files found in {emb_dir}")
        return

    all_mfa_files = sorted(Path(args.mfa_dir).glob("*.npy"))
    print(f"Found {len(emb_files)} speaker embeddings.")
    total_saved = 0
    for emb_file in emb_files:
        speaker_suffix = emb_file.stem
        print(f"\n=== Processing speaker: {speaker_suffix} ===")
        try:
            speaker_emb = np.load(emb_file)
        except Exception as e:
            print(f"❌ Error loading embedding {emb_file}: {e}")
            continue
        if speaker_emb.shape[0] != 192:
            print(f"❌ Error: Expected 192-dim ECAPA embedding, got {speaker_emb.shape[0]}")
            continue
        # Collect input files for this speaker
        mfa_files = [f for f in all_mfa_files if f.stem.endswith(f"_{speaker_suffix}")]
        f0_files = [Path(args.f0_dir) / f.name for f in mfa_files]
        if not mfa_files:
            print(f"❌ No MFA files found for speaker {speaker_suffix}")
            continue
        missing_f0 = [f for f in f0_files if not f.exists()]
        if missing_f0:
            print(f"❌ Error: {len(missing_f0)} F0 files missing for speaker {speaker_suffix}")
            for f in missing_f0[:5]:
                print(f"   Missing: {f}")
            continue
        print(f"Found {len(mfa_files)} files to process for {speaker_suffix}")
        # Run inference
        wavs, basenames = infer_batch(
            model,
            [str(f) for f in mfa_files],
            [str(f) for f in f0_files],
            speaker_emb,
            args.device,
            args.inference_noise_scale,
        )
        # Save waveforms
        saved_count = 0
        for wav, basename in zip(wavs, basenames):
            if wav is None:
                continue
            out_path = Path(args.out_dir) / f"{basename}.wav"
            sf.write(out_path, wav, ap.sample_rate)
            saved_count += 1
        print(f"✅ Done! Saved {saved_count}/{len(wavs)} waveforms for {speaker_suffix} to {args.out_dir}")
        total_saved += saved_count
    print(f"\n=== All done! Total waveforms saved: {total_saved}")


if __name__ == "__main__":
    main()
