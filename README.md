# Local Scribe

Local speech-to-text transcription using Qwen ASR on Apple Silicon via MLX.

## Setup

```bash
uv sync
```

## Usage

```bash
# Record and transcribe (interactive device selection)
uv run python qwen_asr.py
```

Configure via `settings.yaml`:

```yaml
output: recording.wav
language: English
sample_rate: 16000
level_bar_width: 30
transcribe_only: null  # Set to audio file path, or {mic: file1.wav, system: file2.wav}

system_audio:
  enabled: false              # Set to true to capture system audio alongside mic
  device: null                # Device name substring to match (defaults to "blackhole")
  output: recording_system.wav
  gain: 1.0                   # Multiplier for system audio volume (e.g. 1.5 to boost 50%)

diarization:
  enabled: false         # Set to true to identify speakers
  backend: sortformer    # sortformer (default, MLX-native) or pyannote
  model: mlx-community/diar_sortformer_4spk-v1-fp32
  threshold: 0.5         # Activity detection sensitivity (0-1)
  min_duration: 0.0      # Minimum segment length in seconds
  merge_gap: 0.0         # Max gap to merge adjacent segments
  # Pyannote-only settings (used when backend: pyannote)
  pyannote_model: pyannote/speaker-diarization-community-1
  min_speakers: null
  max_speakers: null

summarization:
  enabled: false       # Set to true to enable AI summarization
  model: mlx-community/LFM2-2.6B-Transcript-4bit
  summary_type: executive  # executive, detailed, action_items, key_decisions, participants, topics
```

## Speaker Diarization Setup

Speaker diarization identifies and labels different speakers in the audio (up to 4 speakers).

The default backend is **Sortformer** — an MLX-native model that requires zero configuration:

```yaml
diarization:
  enabled: true
```

### Alternative: pyannote backend

If you need more than 4 speakers or prefer pyannote, you can use it as an alternative backend:

1. Install the optional dependency: `uv sync --extra pyannote`
2. Create a [HuggingFace account](https://huggingface.co/join) and generate an [access token](https://huggingface.co/settings/tokens)
3. Accept the model terms at [pyannote/speaker-diarization-community-1](https://huggingface.co/pyannote/speaker-diarization-community-1)
4. Create a `.env` file with your token (see `.env.example`):
   ```
   HF_TOKEN=your_token_here
   ```
5. Set the backend in `settings.yaml`:
   ```yaml
   diarization:
     enabled: true
     backend: pyannote
   ```

## System Audio Capture (Optional)

Capture system audio (e.g. from a meeting app) alongside your microphone. Disabled by default.

### Setup

1. Install [BlackHole](https://existential.audio/blackhole/) (2ch):
   ```bash
   brew install blackhole-2ch
   # Reboot required after install
   ```

2. Open **Audio MIDI Setup** (in /Applications/Utilities):
   - Click `+` bottom-left → "Create Multi-Output Device"
   - Check both your speakers/headphones **and** BlackHole 2ch
   - Ensure your speakers are listed first

3. Set the Multi-Output Device as your system output in **System Settings → Sound**

4. Enable in `settings.yaml`:
   ```yaml
   system_audio:
     enabled: true
   ```

The tool auto-detects BlackHole by name. If using a different virtual audio device, set `device` to a substring of its name.

### Volume adjustment

macOS may lower system audio volume when an app opens a mic input stream. If system audio sounds quieter than expected, increase `gain` in settings (e.g. `1.5` to boost 50%).

### Output

When enabled, mic and system audio are recorded and transcribed separately, then merged chronologically with `[mic]`/`[system]` source tags.

## Features

- Interactive audio device selection
- Real-time recording level display
- System audio capture alongside mic via BlackHole or similar virtual devices
- Transcripts saved as markdown with YAML frontmatter in `transcripts/YYYY-MM-DD/`
- Optional speaker diarization using Sortformer (MLX-native) or pyannote
- Optional AI summarization using LFM2
- Models cached locally in `models/` directory
