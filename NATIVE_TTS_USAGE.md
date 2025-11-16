# Native TTS Usage Guide

This guide explains how to train Native TTS and use it to generate ground truth datasets for Accent Conversion (AC) training.

## Overview

Native TTS is used in a multi-phase workflow:

1. **Phase 1: Train Native TTS** on native-accented synthetic data (LJSpeech + VCTK)
2. **Phase 2: Generate Ground Truth** using trained Native TTS on non-native data (ARCTIC)
3. **Phase 3: Train AC Transformer** to map non-native → native using ground truth pairs

This guide covers **Phase 1** and **Phase 2**.

## Runpod Setup

Install system dependencies:
```bash
pod# apt-get update && apt-get upgrade -y
pod# apt-get install -y --no-install-recommends \
    gcc g++ make python3 python3-dev \
    espeak-ng libsndfile1-dev libc-dev \
    screen tree
```

Start screen session
```bash
pod# screen -S session
  # Detach: Ctrl + A, then D
  # Reattach: screen -r session
```

To avoid the network volume being the bottleneck, copy the data to the SSD.
```bash
pod# cp -R /dataset/augmented_data /workspace/

# Try this next time
cd /dataset/augmented_data
find . -type f | parallel -j 32 'mkdir -p /workspace/augmented_data/$(dirname {}) && cp {} /workspace/augmented_data/{}'
```

Clone the forked coqui repository with the accent_changer branch:
```bash
pod# cd /workspace
pod# git clone -b accent_changer https://github.com/sangorrin/coqui-ai-TTS.git
pod# cd coqui-ai-TTS
```

Install Coqui python dependencies:
```bash
pod# uv venv /workspace/accent_changer --system-site-packages
pod# source /workspace/accent_changer/bin/activate
pod# uv pip install -e .[all]
```

## Phase 1: Training Native TTS

The script automatically:
- Calculates `num_chars` from `phoneme_map.json`
- Configures batch size based on GPU VRAM (default: auto-detect)
- Sets up multi-speaker training with ECAPA embeddings

```bash
pod(accent_changer)# cd /workspace/coqui-ai-TTS/recipes/ljspeech/vits_tts
pod(accent_changer)# python train_native_tts.py \
  --data_path /workspace/augmented_data \
  --vram 24 \
  --use_transfer_learning
```

Checkpoints and logs saved to:
```bash
ls /workspace/coqui-ai-TTS/recipes/ljspeech/vits_tts/native_tts_ljspeech_freevc_vctk-{date}/
  ├── checkpoint_*.pth     # Model checkpoints
  ├── config.json          # Training config
  ├── events.out.tfevents  # TensorBoard logs
  └── ...
```

Intermediate monitoring
```bash
pod(accent_changer)# ln -s /workspace/coqui-ai-TTS/recipes/ljspeech/vits_tts/native_tts_ljspeech_freevc_vctk-November-16-2025_01+59AM-b35930a9/best_model.pth checkpoint.pth
pod(accent_changer)# ln -s /workspace/coqui-ai-TTS/recipes/ljspeech/vits_tts/native_tts_ljspeech_freevc_vctk-November-16-2025_01+59AM-b35930a9/config.json config.json

# Get some random ARCTIC inputs for monitoring the quality of the generated audio
pod(accent_changer)# python << 'EOF'
import random
from pathlib import Path
import shutil

# Source directories
phones_dir = Path('/dataset/arctic_data/mfa_alignments')
f0_dir = Path('/dataset/arctic_data/f0_features')
emb_dir = Path('/dataset/arctic_data/speaker_embeddings')
wavs_dir = Path('/dataset/arctic_data/wavs_16k')
out_dir = Path('/workspace/arctic_infer_samples')

(out_dir / 'mfa_alignments').mkdir(parents=True, exist_ok=True)
(out_dir / 'f0_features').mkdir(exist_ok=True)
(out_dir / 'speaker_embeddings').mkdir(exist_ok=True)
(out_dir / 'wavs_16k').mkdir(exist_ok=True)

# Get all utterance files (excluding phoneme_map)
utt_files = [f for f in phones_dir.glob('*.npy') if f.stem != 'phoneme_map']
sampled = random.sample(utt_files, min(10, len(utt_files)))

print(f'Preparing {len(sampled)} ARCTIC samples for inference in {out_dir}\n')

for ph_file in sampled:
    utt = ph_file.stem
    f0_file = f0_dir / f'{utt}.npy'
    wav_file = wavs_dir / f'{utt}.wav'
    # Copy phoneme and f0 files
    shutil.copy(ph_file, out_dir / 'mfa_alignments' / f'{utt}.npy')
    if f0_file.exists():
        shutil.copy(f0_file, out_dir / 'f0_features' / f'{utt}.npy')
    else:
        print(f'Warning: F0 file missing for {utt}')
    # Copy input wav for comparison
    if wav_file.exists():
        shutil.copy(wav_file, out_dir / 'wavs_16k' / f'{utt}.wav')
    else:
        print(f'Warning: Input wav missing for {utt}')
    # Copy speaker embedding (by speaker suffix)
    spk = utt.split('_')[-1]
    emb_file = emb_dir / f'{spk}.npy'
    if emb_file.exists():
        shutil.copy(emb_file, out_dir / 'speaker_embeddings' / f'{spk}.npy')
    else:
        print(f'Warning: Speaker embedding missing for {spk}')

print('Done.')
EOF

pod(accent_changer)# python infer_native_tts.py \
  --checkpoint /workspace/checkpoint.pth \
  --config /workspace/config.json \
  --mfa-dir /workspace/arctic_infer_samples/mfa_alignments \
  --f0-dir /workspace/arctic_infer_samples/f0_features \
  --speaker-emb-dir /workspace/arctic_infer_samples/speaker_embeddings \
  --out-dir /workspace/arctic_infer_samples/native_generated \
  --device cuda

local# rsync -avz  runpod-1:/workspace/arctic_infer_samples .
local# cd arctic_infer_samples
local# afplay wavs_16k/arctic_b0511_ASI.wav
local# afplay native_generated/arctic_b0511_ASI.wav # check audio quality
```

Estimations:
- **RTX 4090**: ~24-48 hours for 100k steps
- **Convergence**: Monitor mel reconstruction loss and audio quality
- **Recommended**: Train for at least 100k steps

---

## Phase 2: Generate Ground Truth from ARCTIC

After training Native TTS on native-accented data, use it to generate native-accented audio from non-native ARCTIC speakers. This creates paired training data for the Accent Conversion transformer:
- **Input**: Non-native ARCTIC audio (original recordings)
- **Output**: Native-accented audio (same speaker identity, native pronunciation)

### Running Inference

```bash
cd /workspace/coqui-ai-TTS/recipes/ljspeech/vits_tts

# Process all ARCTIC speakers
python infer_native_tts.py \
  --checkpoint /dataset/checkpoint.pth \
  --config /dataset/config.json \
  --mfa-dir /dataset/arctic_data/mfa_alignments \
  --f0-dir /dataset/arctic_data/f0_features \
  --speaker-emb-dir /dataset/arctic_data/speaker_embeddings \
  --out-dir /dataset/arctic_data/native_generated \
  --device cuda
```

Paired Dataset Structure
```
/dataset/arctic_data/
├── wavs_16k/                    # Non-native (original ARCTIC)
│   ├── arctic_a0001_ABA.wav
│   └── ...
└── native_generated/            # Native-accented (generated by Native TTS)
    ├── arctic_a0001_ABA.wav
    └── ...
```

## Next Steps

After generating ground truth:

1. **Phase 3**: Train AC Transformer
   - Input: Non-native ARCTIC audio
   - Target: Native-accented generated audio
   - Architecture: Transformer-based sequence-to-sequence model
   - Objective: Learn to map non-native acoustic features → native-accented features

2. **Evaluation**:
   - MOS (Mean Opinion Score) for naturalness
   - Accent classification accuracy
   - Speaker similarity (cosine distance of ECAPA embeddings)
   - Pronunciation accuracy (phoneme error rate)

---

## References

- **Native TTS Paper**: https://arxiv.org/abs/2506.16580
- **ac_playground preprocessing**: https://github.com/sangorrin/ac_playground
- **ARCTIC Dataset**: http://www.festvox.org/cmu_arctic/
- **Montreal Forced Aligner**: https://mfa-models.readthedocs.io/
