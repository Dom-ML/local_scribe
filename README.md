# Local Scribe

Local speech-to-text transcription using Qwen ASR on Apple Silicon via MLX.

## Setup

```bash
uv sync
```

## Usage

```bash
# List available audio input devices
uv run python qwen_asr.py --list-devices

# Record and transcribe (10 seconds default)
uv run python qwen_asr.py -d <device_index>

# Record with custom duration
uv run python qwen_asr.py -d 0 -t 30

# Transcribe an existing audio file
uv run python qwen_asr.py -T audio.wav
```

## Options

| Option | Description |
|--------|-------------|
| `--list-devices` | List available audio input devices |
| `-d, --device` | Audio input device index |
| `-t, --duration` | Recording duration in seconds (default: 10) |
| `-o, --output` | Output audio file path (default: recording.wav) |
| `-l, --language` | Transcription language (default: English) |
| `-T, --transcribe-only` | Transcribe existing audio file |
