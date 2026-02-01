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
transcribe_only: null  # Set to audio file path to transcribe existing file

diarization:
  enabled: false       # Set to true to identify speakers
  model: pyannote/speaker-diarization-community-1
  min_speakers: null   # Optional: minimum number of speakers
  max_speakers: null   # Optional: maximum number of speakers

summarization:
  enabled: false       # Set to true to enable AI summarization
  model: mlx-community/LFM2-2.6B-Transcript-4bit
  summary_type: executive  # executive, detailed, action_items, key_decisions, participants, topics
```

## Speaker Diarization Setup

Speaker diarization identifies and labels different speakers in the audio. To enable:

1. Create a [HuggingFace account](https://huggingface.co/join) and generate an [access token](https://huggingface.co/settings/tokens)

2. Accept the model terms at [pyannote/speaker-diarization-community-1](https://huggingface.co/pyannote/speaker-diarization-community-1)

3. Create a `.env` file with your token (see `.env.example`):
   ```
   HF_TOKEN=your_token_here
   ```

4. Enable in `settings.yaml`:
   ```yaml
   diarization:
     enabled: true
   ```

## Features

- Interactive audio device selection
- Real-time recording level display
- Transcripts saved as markdown with YAML frontmatter in `transcripts/YYYY-MM-DD/`
- Optional speaker diarization using pyannote
- Optional AI summarization using LFM2
- Models cached locally in `models/` directory
