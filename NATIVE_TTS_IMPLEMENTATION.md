# Native TTS Implementation for Coqui TTS

## Overview

This implementation adds Native TTS support to Coqui-TTS based on the paper: https://arxiv.org/abs/2506.16580 (Streaming Non-Autoregressive Model for Accent Conversion and Pronunciation Improvement).

Native TTS was initially implemented by modifying `vits.py`, but was later separated into standalone files (`native_tts.py`, `native_tts_config.py`) to keep the original VITS implementation intact. The architecture is derived from VITS with specific modifications for accent conversion using precomputed MFA alignments, F0 features, and speaker embeddings.

## Key Architectural Differences from VITS

### What's Different

1. **Prior Encoder (replaces TextEncoder + Duration Predictor)**
   - New `PriorEncoder` class that conditions on F0
   - Accepts MFA-aligned phoneme IDs (precomputed at 20ms frames)
   - F0 conditioning via learned Conv1d(1→1) for normalization
   - Concatenates phoneme embeddings (192D) + F0 embedding (1D) → 193D
   - No duration predictor needed (MFA provides timing)

2. **No Monotonic Alignment Search (MAS)**
   - MFA provides pre-aligned phonemes at 20ms frames
   - 1:1 correspondence between phoneme frames and spectrogram frames
   - Removes need for learned alignment

3. **Required Speaker Embeddings**
   - VITS: Optional speaker embeddings (learned or d-vectors)
   - Native TTS: **Required** 192-dim ECAPA-TDNN embeddings (precomputed)
   - One embedding per speaker (not per utterance)

4. **Audio Configuration**
   - Sample rate: 22050 Hz (VITS) → **16000 Hz** (Native TTS)
   - Hop length: 256 samples (VITS) → **320 samples** (Native TTS = 20ms frames)
   - Upsample rates: [8,8,2,2] → **[8,8,5,1]** (product: 256→320)
   - Upsample kernels: [16,16,4,4] → **[16,16,10,2]**
   - Uses linear spectrograms (513 channels) for posterior encoder

5. **No Text Processing**
   - VITS: Text → phonemes → embeddings
   - Native TTS: Precomputed MFA phoneme IDs → embeddings (no cleaners/tokenizers)

### What's the Same

- Posterior encoder architecture (16-layer WaveNet on linear spec)
- Flow architecture (4 Residual Coupling Blocks)
- HiFi-GAN decoder (resblock type "1", same kernel/dilation configs except upsample rates)
- Discriminator (Multi-Period Discriminator with periods [2,3,5,7,11])
- Training losses (KL divergence, mel reconstruction, adversarial)
- Noise scales (1.0 training, 0.667 inference)

## Implementation Files

### Created (derived from VITS)

1. **`TTS/tts/models/native_tts.py`** - Complete Native TTS implementation
   - `NativeTTSArgs` - Model arguments (based on VitsArgs)
   - `NativeTTSAudioConfig` - Audio configuration (16kHz, 320 hop length)
   - `NativeTTSDataset` - Custom dataset loader (no text processing)
   - `NativeTTS` - Main model class (based on Vits)

2. **`TTS/tts/configs/native_tts_config.py`** - Training configuration
   - `NativeTTSConfig` - Based on VitsConfig
   - No `test_sentences` (would require MFA/F0 preprocessing pipeline)

3. **`recipes/ljspeech/vits_tts/train_native_tts.py`** - Training script
   - `calculate_num_chars()` - Reads phoneme vocabulary from MFA phoneme_map.json
   - `native_tts_formatter()` - Loads precomputed artifacts (MFA/F0/ECAPA)
   - Auto-configures batch size based on GPU VRAM

### Modified

4. **`TTS/tts/layers/vits/networks.py`** - Added PriorEncoder class
   - New class specifically for Native TTS
   - Does NOT modify existing TextEncoder (VITS remains intact)

## Model Arguments (NativeTTSArgs)

All parameters are documented inline in `native_tts.py` showing their relationship to VitsArgs:

**Inherited unchanged from VITS:**
- `out_channels=513`, `hidden_channels=192`, `spec_segment_size=32`
- All posterior encoder params (kernel=5, dilation=1, layers=16)
- All flow params (kernel=5, dilation=1, **layers=4**)
- Resblock configs (type="1", kernels=[3,7,11], dilations=[[1,3,5]×3])
- Discriminator params (periods=[2,3,5,7,11])
- Noise scales (1.0, 0.667)

**Modified from VITS:**
- `upsample_rates_decoder`: [8,8,2,2] → [8,8,5,1] (for 320 hop length)
- `upsample_kernel_sizes_decoder`: [16,16,4,4] → [16,16,10,2]

**Renamed from VITS (same values):**
- Prior encoder params (was text_encoder_*):
  - `hidden_channels_ffn_prior=768` (was hidden_channels_ffn_text_encoder)
  - `num_heads_prior=2` (was num_heads_text_encoder)
  - `num_layers_prior=6` (was num_layers_text_encoder)
  - `kernel_size_prior=3` (was kernel_size_text_encoder)
  - `dropout_p_prior=0.1` (was dropout_p_text_encoder)

**New for Native TTS:**
- `f0_embedding_dim=1` - Scalar F0 conditioning (VQMIVC [23] default approach)
- `embedded_speaker_dim=192` - Fixed ECAPA-TDNN dimension

**Removed from VITS:**
- Stochastic Duration Predictor params (use_sdp, noise_scale_dp, etc.)
- Learned speaker embeddings (use_speaker_embedding, num_speakers, etc.)
- Language embedding params
- Speaker encoder loss params
- Freeze flags, encoder_sample_rate, interpolate_z

## Data Requirements

### Preprocessing Pipeline (ac_playground)

Complete workflow documented at: https://github.com/sangorrin/ac_playground/blob/main/PREPARE_AUGMENTED_DATA.md

**Step 1: Augment Native Audio Utterances**
- Run `freevc_batch.py` to convert LJSpeech text with VCTK speakers
- Creates native-accented speech: `LJ001-0001_p225.wav` (LJSpeech text, VCTK speaker voice)
- Resample to 16kHz mono using `resample_to_16k.py`
- Output: `/workspace/augmented_data/wavs_16k/`

**Step 2: F0 Extraction (20ms frames, YAAPT)**
- Run `f0_20ms_batch.py` on 16kHz audio
- Uses YAAPT algorithm (amfm_decompy.pYAAPT)
- Output: `/workspace/augmented_data/f0_features/*.npy` (float32, Hz values, NaN=unvoiced)

**Step 3: MFA Alignment (20ms frames)**
- Prepare corpus: `mfa_prepare.py` creates MFA corpus structure with `.wav` and `.lab` files
- Run MFA + upsample: `mfa_upsample_batch.py`
  - Runs Montreal Forced Aligner with english_us_mfa dictionary and english_mfa acoustic model
  - Upsamples TextGrid alignments to 20ms frames
  - Generates `phoneme_map.json` in output directory
- Fix length mismatches: `fix_phones_lengths.py` pads phoneme vectors to match F0 lengths
- Output: `/workspace/augmented_data/mfa_alignments/*.npy` (int16 phoneme IDs) + `phoneme_map.json`

**Step 4: Speaker Embeddings (ECAPA-TDNN)**
- Run `speaker_embed_batch.py` on VCTK reference files (one 16kHz file per speaker)
- Uses ECAPA-TDNN from SpeechBrain (speechbrain/spkrec-ecapa-voxceleb)
- Output: `/workspace/augmented_data/speaker_embeddings/<speaker_id>.npy` (192-dim float32)

**Step 5: Sanity Checks**
- Run `check_features_20ms.py` to verify all artifacts match lengths

### Dataset Directory Structure

Based on `train_native_tts.py` formatter and ac_playground preprocessing:

```
/workspace/augmented_data/    # <-- This is --data_path for train_native_tts.py
  ├── wavs_16k/                # Audio files (NOTE: wavs_16k NOT wavs_16khz)
  │   └── LJ001-0001_p225.wav  # LJSpeech ID + VCTK speaker
  ├── mfa_alignments/          # Phoneme IDs at 20ms frames
  │   ├── LJ001-0001_p225.npy  # int16 array [T] with phoneme IDs
  │   └── phoneme_map.json     # {"sil": 0, "AA": 1, "AE": 2, ...}
  ├── f0_features/             # F0 values at 20ms frames
  │   └── LJ001-0001_p225.npy  # float32 array [T] with Hz values (NaN=unvoiced)
  └── speaker_embeddings/      # ECAPA embeddings (one per speaker)
      └── p225.npy             # float32 array [192] speaker embedding
```

**File naming:**
- Audio/MFA/F0: `<basename>_<speaker_id>.{wav,npy}` where basename = LJSpeech ID
- Speaker embeddings: `<speaker_id>.npy` (one per speaker, extracted from audio filename)
- Example: `LJ001-0001_p225.wav` → speaker_id = `p225`

### File Format Details

- **MFA files**: `np.int16`, shape `[T]`, values 0 to num_chars-1
- **F0 files**: `np.float32`, shape `[T]`, Hz values (80-400 typical, NaN=unvoiced)
- **Speaker files**: `np.float32`, shape `[192]`, ECAPA-TDNN embedding
- **Critical**: MFA and F0 arrays MUST have matching lengths (validated in dataset)

## Training

### Configuration

The training script (`train_native_tts.py`) automatically:
1. Calculates `num_chars` from `phoneme_map.json` (max_id + 1)
2. Configures batch size based on GPU VRAM
3. Sets up the dataset formatter

```bash
cd recipes/ljspeech/vits_tts
python train_native_tts.py --data_path /path/to/dataset --vram 24
```

### Key Training Details

- **No text processing**: Dataset loads `.npy` files directly
- **Phoneme vocabulary**: Dynamically calculated from MFA phoneme_map.json
- **Pad token**: ID 0 (typically 'sil' in MFA)
- **Multi-speaker**: ~65k utterances (LJSpeech text × VCTK speakers via FreeVC)
- **Validation**: Dataset checks F0/MFA length match, raises error on mismatch

### Forward Pass

**Training:**
1. Prior encoder: MFA phonemes + F0 → prior distribution p(z|phonemes,F0,speaker)
2. Posterior encoder: Linear spec + speaker → posterior q(z|spec,speaker)
3. Flow: Transform posterior to match prior
4. Decoder: Random segment of z → waveform
5. Losses: KL divergence + mel reconstruction + adversarial

**Inference:**
1. Prior encoder: MFA phonemes + F0 + speaker → prior p(z)
2. Sample z from prior with noise_scale=0.667
3. Inverse flow: z → refined latent
4. Decoder: Full sequence → native-accented waveform
5. Preserves F0/duration/speaker, improves pronunciation

## Architecture Comparison: VITS vs Native TTS

| Component | Standard VITS | Native TTS |
|-----------|--------------|------------|
| **Text Input** | Raw text → G2P → phonemes | Precomputed MFA phoneme IDs |
| **Alignment** | Learned (MAS during training) | Precomputed (MFA at 20ms) |
| **Duration** | Predicted by Duration Predictor | From MFA (no prediction) |
| **Conditioning** | Optional language/speaker | **Required** F0 + ECAPA speaker |
| **Prior Encoder** | TextEncoder (phoneme→latent) | PriorEncoder (phoneme+F0→latent) |
| **F0** | Implicit in latent space | **Explicit** conditioning |
| **Speaker** | Optional (learned or d-vector) | **Required** (192-dim ECAPA) |
| **Sample Rate** | 22050 Hz (default) | **16000 Hz** |
| **Hop Length** | 256 samples (~11.6ms) | **320 samples (20ms)** |
| **Upsample Rates** | [8, 8, 2, 2] (product=256) | **[8, 8, 5, 1]** (product=320) |
| **Flow Layers** | 4 Residual Coupling Blocks | 4 (same) |
| **Posterior Input** | Mel or linear spec | **Linear spec** (513 channels) |
| **Use Case** | General TTS | **Accent conversion** |

## Implementation Notes

### F0 Embedding Design

- **Dimension**: 1 (scalar, follows VQMIVC [23] default)
- **Projection**: Conv1d(1→1) learns scale and bias for normalization
- **Why keep Conv1d with dim=1?**: F0 is in Hz (80-400), phoneme embeddings are N(0, 0.072). The learned weight/bias brings them into compatible scales.
- **Alternative**: Could remove Conv1d and concatenate raw F0, but loses automatic normalization

### Phoneme Vocabulary

- **NOT hardcoded**: `num_chars` calculated from `phoneme_map.json` at training time
- **Default value**: 100 (placeholder in NativeTTSArgs, overridden by training script)
- **Pad token**: ID 0 (typically 'sil' silence phoneme)
- **Typical size**: 40-80 phonemes depending on MFA dictionary

### Why Separate from VITS?

1. **Code clarity**: Different dataset, different preprocessing, different use case
2. **Maintainability**: VITS can be updated without breaking Native TTS
3. **Flexibility**: Can diverge architectures independently
4. **Testing**: Each model has its own tests and validation

## References

- **Native TTS Paper**: https://arxiv.org/abs/2506.16580
  "Streaming Non-Autoregressive Model for Accent Conversion and Pronunciation Improvement"
- **VQMIVC [23]**: https://arxiv.org/abs/2106.10280
  Referenced for F0 encoder approach (default: dim_lf0=1, scalar F0)
- **VITS**: https://arxiv.org/abs/2106.06103
  Base architecture for Native TTS
- **Preprocessing Scripts**: https://github.com/sangorrin/ac_playground
  MFA alignment, F0 extraction, speaker embedding extraction
- **Montreal Forced Aligner (MFA)**: https://mfa-models.readthedocs.io/
  Phoneme alignment tool
- **YAAPT**: Yet Another Algorithm for Pitch Tracking
  F0 extraction algorithm (amfm_decompy.pYAAPT)
- **ECAPA-TDNN**: SpeechBrain speaker encoder
  Model: speechbrain/spkrec-ecapa-voxceleb (192-dim embeddings)

## Changelog

- **Initial Implementation**: Started as modifications to vits.py
- **Separation**: Moved to standalone native_tts.py to keep VITS intact
- **Current Status**: Complete implementation with documented parameter origins
