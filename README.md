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

summarization:
  enabled: false       # Set to true to enable AI summarization
  model: mlx-community/LFM2-2.6B-Transcript-4bit
  summary_type: executive  # executive, detailed, action_items, key_decisions, participants, topics
```

## Features

- Interactive audio device selection
- Real-time recording level display
- Transcripts saved as markdown with YAML frontmatter in `transcripts/YYYY-MM-DD/`
- Optional AI summarization using LFM2
- Models cached locally in `models/` directory
