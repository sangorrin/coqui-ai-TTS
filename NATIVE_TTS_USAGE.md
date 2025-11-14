# Native TTS Usage Guide

This guide explains how to train Native TTS and use it to generate ground truth datasets for Accent Conversion (AC) training.

## Overview

Native TTS is used in a multi-phase workflow:

1. **Phase 1: Train Native TTS** on native-accented synthetic data (LJSpeech + VCTK)
2. **Phase 2: Generate Ground Truth** using trained Native TTS on non-native data (ARCTIC)
3. **Phase 3: Train AC Transformer** to map non-native → native using ground truth pairs

This guide covers **Phase 1** and **Phase 2**.

## Runpod Setup

Deploy a Runpod instance with an RTX 4090 GPU and attach the network
volume at `/dataset` by modifying the template. Set the pod SSD disk
size to 200GB. SSH into the instance and copy the augmented data (13GB) from
the network volume (`/dataset/augmented_data`) to the SSD (`/workspace`).
It takes about 10 minutes.
```bash
pod# ssh runpod-1
pod# cd /dataset
pod# cp -R augmented_data /workspace
```

Install system dependencies:
```bash
pod# apt-get update && apt-get upgrade -y
pod# apt-get install -y --no-install-recommends \
    gcc g++ make python3 python3-dev \
    espeak-ng libsndfile1-dev libc-dev \
    screen tree
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

Start screen session and activate the environment.
```bash
pod# screen -S session
  # Detach: Ctrl + A, then D
  # Reattach: screen -r session
pod# source /workspace/accent_changer/bin/activate
```

Perform a sanity check
```bash
pod(accent_changer)# cd /workspace
pod(accent_changer)# git clone https://github.com/sangorrin/ac_playground.git
pod(accent_changer)# python /workspace/ac_playground/check_features_20ms.py \
    --wav-dir /workspace/augmented_data/wavs_16k \
    --phones-dir /workspace/augmented_data/mfa_alignments \
    --f0-dir /workspace/augmented_data/f0_features \
    --spk-embeds-dir /workspace/augmented_data/speaker_embeddings \
    --report /workspace/sanity_report.csv \
    --limit 0 \
    --workers 32
```

## Phase 1: Training Native TTS

### Training Command

```bash
pod(accent_changer)# cd /workspace/coqui-ai-TTS/recipes/ljspeech/vits_tts

pod(accent_changer)# python train_native_tts.py \
  --data_path /workspace/augmented_data \
  --vram 24 \
  --use_transfer_learning # 300 -> 100 epochs, 72 -> 24 hours, 115 -> 40 start mel loss
```

### Training Configuration

The script automatically:
- Calculates `num_chars` from `phoneme_map.json`
- Configures batch size based on GPU VRAM (default: auto-detect)
- Sets up multi-speaker training with ECAPA embeddings

**Key hyperparameters** (edit in `train_native_tts.py` if needed):
- `batch_size`: Auto-configured based on VRAM
- `eval_batch_size`: Half of batch_size
- `num_workers`: Auto-configured based on CPU cores
- `lr_gen`: 2e-4 (generator learning rate)
- `lr_disc`: 2e-4 (discriminator learning rate)

### Monitoring Training

Checkpoints and logs saved to:
```
/workspace/coqui-ai-TTS/recipes/ljspeech/vits_tts/native_tts_ljspeech_freevc_vctk-{date}/
  ├── checkpoint_*.pth     # Model checkpoints
  ├── config.json          # Training config
  ├── events.out.tfevents  # TensorBoard logs
  └── ...
```

View training progress with TensorBoard:
```bash
tensorboard --logdir recipes/ljspeech/vits_tts/
```

### Training Duration

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
for speaker_file in /dataset/arctic_data/speaker_embeddings/*.npy; do
  python infer_native_tts.py \
    --checkpoint /dataset/checkpoint.pth \
    --config /dataset/config.json \
    --mfa-dir /dataset/arctic_data/mfa_alignments \
    --f0-dir /dataset/arctic_data/f0_features \
    --speaker-emb "$speaker_file" \
    --out-dir /dataset/arctic_data/native_generated \
    --device cuda
done
```

**Output**: Generated files will be saved with speaker suffixes (e.g., `arctic_a0001_ABA.wav`, `arctic_a0001_ASI.wav`) in `/dataset/arctic_data/native_generated/`.

### Paired Dataset Structure

```
/dataset/arctic_data/
├── wavs_16k/                    # Non-native (original ARCTIC)
│   ├── arctic_a0001_ABA.wav
│   └── ...
└── native_generated/            # Native-accented (generated by Native TTS)
    ├── arctic_a0001_ABA.wav
    └── ...
```

This paired dataset will be used in Phase 3 to train the Accent Conversion transformer.

---

## Troubleshooting

### Common Issues

**1. Length mismatch between MFA and F0**

Error: `MFA length 245 != F0 length 246`

Solution: Run `fix_phones_lengths.py` to pad phoneme vectors:
```bash
python fix_phones_lengths.py \
  --phones-dir ARCTIC_mfa_phones_20ms \
  --f0-dir ARCTIC_f0_features \
  --out-dir ARCTIC_mfa_alignments \
  --workers 32
```

**2. Speaker embedding dimension mismatch**

Error: `Expected 192-dim ECAPA embedding, got 512`

Solution: Use ECAPA-TDNN from SpeechBrain (not other models):
```python
# In speaker_embed_batch.py, ensure:
from speechbrain.pretrained import EncoderClassifier
classifier = EncoderClassifier.from_hparams(
    source="speechbrain/spkrec-ecapa-voxceleb",
    savedir="pretrained_models/spkrec-ecapa-voxceleb"
)
```

**3. Out of memory during inference**

Solution: Process files one at a time (current implementation) or reduce model to CPU:
```bash
python infer_native_tts.py ... --device cpu
```

**4. Phoneme vocabulary mismatch**

Error: `Index out of range in embedding layer`

Solution: Ensure the same MFA dictionary was used for training and inference. Check `phoneme_map.json` in both datasets.

---

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
