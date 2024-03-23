# Pytranscript 🎙️

Pytranscript is a powerful Python library and command-line tool designed to seamlessly convert video or audio files into text and translate them into various languages. It acts as a simple yet effective wrapper around [Vosk](https://alphacephei.com/vosk/), [ffmpeg](https://ffmpeg.org/), and [deep-translator](https://pypi.org/project/deep-translator/), making the transcription and translation process straightforward.

## Prerequisites

Before using pytranscript, ensure you have the following dependencies installed:

- [ffmpeg](https://ffmpeg.org/download.html) for audio conversion.
- [vosk-models](https://alphacephei.com/vosk/models) required for speech recognition. You will have to specify to your specific model path in the `--model` argument.

## Installation

```bash
pip install pytranscript
```

## Usage

### Command Line

```bash
pytranscript INPUT_FILE [OPTIONS]
```

### Options

- `-m, --model` - Path to the Vosk model directory. Always required.
- `-o, --output` - Output file where the text will be saved. Default: input file name with `.txt` extension.
- `-li, --lang_input` - Language of the input / the model. Default: auto.
- `-lo --lang_input` - Language to translate the text to. Default: no translation.
- `-s, --start` - Start time of the audio to transcribe in seconds.
- `-e, --end` - End time of the audio to transcribe in seconds.
- `--max_size` - Will stop the transcription if the output file reaches the specified size in bytes. Takes precedence over the `--end` option.
- `--keep-wav` - Keep the converted audio wav file after the process is done.
- `-v, -verbosity` - Verbosity level. 0: no output, 1: only errors, 2: errors, info and progressbar, 3: debug. Default: 2.

## Example

The most basic usage is:

```bash
pytranscript video.mp4 -m vosk-model-en-us-aspire-0.2 -lo fr
```

Where `vosk-model-en-us-aspire-0.2` is the Vosk model directory. The text will be translated from English to French, and the output will be saved in a file named `video.txt`.

Using the `keep-wav` option can be useful if you want to do many transcriptions within the same file, allowing you to use the same `.wav` file for each transcription, thus saving conversion time.
 ⚠️ The `.wav` file is cropped according to the start and end time options.

### API

The API provides a Transcript object containing the time and text. The `translate` method can be used to get another Transcript object with the translated text. The output saved in a file in the cli is just the string
`str(transcript)`.

A reproduction of the previous example using the API:

```python
import pytranscript as pt

wav_file = pt.to_valid_wav('video.mp4', "video.wav", start=0, end=None)
transcript = pt.transcribe(wav_file, model='vosk-model-en-us-aspire-0.2', max_size=None)
transcript_fr = transcript.translate('fr')

with open('video.txt', 'w', encoding="utf8") as f:
    f.write(str(transcript_fr))
```
