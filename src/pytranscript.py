from __future__ import annotations

import json
import logging
import typing
import warnings
import wave
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Literal, NamedTuple

import deep_translator
import ffmpeg
import tap  # typed_argument_parser
import vosk
from tqdm import tqdm

if TYPE_CHECKING:
    from os import PathLike
type StrPath = str | PathLike[str]


TranscriptFormat = Literal["csv", "json", "srt", "txt", "vtt"]
TRANSCRIPT_FORMATS: tuple[TranscriptFormat] = typing.get_args(TranscriptFormat)


class LineError(NamedTuple):
    time: float
    line: str
    error: Exception


def seconds_to_time(seconds: float) -> str:
    """Convert seconds to time.

    Args:
        seconds (float): the number of seconds

    Returns:
        str: the time in the format :
            -"mm:ss.ms" if 0 hours
            -"hh:mm:ss" if 0 days
            -"dd hh:mm:ss" if at least 1 day
    """
    seconds = float(seconds)
    if seconds < 3600:  # noqa: PLR2004
        minutes, seconds = divmod(seconds, 60)
        minutes = int(minutes)
        if seconds == int(seconds):
            return f"{minutes:02d}:{seconds:02d}"
        int_part, dec_part = str(seconds).split(".")
        dec_part = dec_part[:2]
        int_part = int(int_part)
        return f"{minutes:02d}:{int(int_part):02d}.{dec_part}"
    if seconds < 86400:  # noqa: PLR2004
        hours, minutes = divmod(seconds, 3600)
        return f"{hours}:{seconds_to_time(minutes)}"
    days, hours = divmod(seconds, 86400)
    hours, minutes = divmod(hours, 3600)
    return f"{days}d {hours:02d}:{seconds_to_time(minutes)}"


def seconds_to_srt_time(seconds: float) -> str:
    """Convert seconds to SRT time format.

    Args:
        seconds (float): the number of seconds

    Returns:
        str: the time in the format "hh:mm:ss,ms"
    """
    seconds = float(seconds)
    hours, remainder = divmod(seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    int_seconds, dec_seconds = str(seconds).split(".")
    dec_seconds = dec_seconds[:3]
    return f"{int(hours):02d}:{int(minutes):02d}:{int(int_seconds):02d},{dec_seconds}"


@dataclass
class Transcript:
    """A transcript of a video or audio file.


    Args:
        time (list[float]): a list of the time of each line in the transcript.
        text (list[str]): a list of the text of each line in the transcript.
        language (str): the language of the transcript. Default: "auto"
        time_end (float): the time of the last line in the transcript. If not specified,
        it will be the time of the last line + 5 seconds.
    """

    time: list[float] = field(default_factory=list)
    text: list[str] = field(default_factory=list)
    language: str = "auto"
    time_end: float | None = None

    @property
    def _time_end(self) -> float:
        if self.time_end is not None:
            return self.time_end
        return self.time[-1] + 5

    def append(self, time: float, text: str) -> None:
        self.time.append(time)
        self.text.append(text)

    def __str__(self):
        return "\n".join(
            f"{seconds_to_time(time)} : {line}"
            for time, line in zip(self.time, self.text, strict=True)
        )

    def __len__(self):
        return len(self.time)

    def __getitem__(self, key):
        return self.time[key], self.text[key]

    def translate(self, target: str) -> tuple[Transcript, list[LineError]]:
        """Return a translated version of the transcript.

        Args:
            target (str, optional): language of the translation.
            For example, "fr" for French, "en" for English, etc.
            source (str, optional): language of the transcript.
            Defaults to "auto".

        Returns:
            - Transcript: the translated transcript
            - list[LineError]: a list of errors that occurred during the translation
        """
        translated = Transcript(time_end=self.time_end)
        errors: list[LineError] = []

        _iter = zip(self.time, self.text, strict=True)
        pbar = tqdm(
            _iter, total=len(self.time), unit_scale=True, unit="line", desc="Translate"
        )
        for time, line in pbar:
            try:
                translator = deep_translator.GoogleTranslator(
                    source=self.language, target=target
                )
                new_line = translator.translate(line)
                translated.append(time, new_line)
            except Exception as e:  # noqa: BLE001
                errors.append(LineError(time, line, e))
        return translated, errors

    def srt_generator(self):
        """Generate the transcript as a string in SRT format, line by line."""

        def one_line(start, end, line):
            start, end = map(seconds_to_srt_time, (start, end))
            return f"{start} --> {end}\n{line}\n\n"

        nb_lines = len(self)
        for i, (time, line) in enumerate(
            zip(self.time, self.text, strict=True), start=1
        ):
            if i == nb_lines:
                end = self._time_end
            else:
                after_time = self.time[i]
                end = min(after_time, time + 5)
            yield one_line(time, end, line)

    def vtt_generator(self):
        """Generate the transcript as a string in VTT format, line by line."""
        yield "WEBVTT\n\n"
        for srt_line in self.srt_generator():
            if " --> " in srt_line:
                yield srt_line.replace(",", ".")
            else:
                yield srt_line

    def csv_generator(self):
        """Generate the transcript as a string in CSV format, line by line."""
        yield "time,text\n"
        for time, line in zip(self.time, self.text, strict=True):
            yield f"{time},{line}\n"

    def to_srt(self) -> str:
        """Return the transcript as a string in SRT format."""
        return "".join(self.srt_generator())

    def to_vtt(self) -> str:
        """Return the transcript as a string in VTT format."""
        return "".join(self.vtt_generator())

    def to_json(self) -> str:
        """Return the transcript as a string in JSON format."""
        return json.dumps({"text": self.text, "time": self.time})

    def to_txt(self) -> str:
        """Return the transcript as a string in TXT format."""
        return str(self)

    def to_csv(self) -> str:
        """Return the transcript as a string in CSV format."""
        return "".join(self.csv_generator())

    def write(self, output: StrPath, format: TranscriptFormat) -> None:  # noqa: A002
        """Write the transcript to a file.

        Args:
            output (Path): the path to the output file.
            format (TranscriptFormat): the format of the transcript.
        """
        method = getattr(self, f"to_{format}")
        Path(output).write_text(method())


def to_valid_wav(
    source: StrPath,
    output: StrPath | None = None,
    start: float = 0,
    end: float | None = None,
) -> Path:
    """Convert a video or audio file to a wav file.

    Args:
        source (Path): the path to the file to convert.
        output (Path, optional): the path to the wav file.
        If not specified, the wav file will be created in the same directory
        and with the same name as the source file.
        start (int, optional): the time in seconds to start the conversion.
        Defaults to 0.
        end (int, bool, optional): the time in seconds to end the conversion.
        If not specified, the conversion will be done until the end of the file.
        name (str, bool, optional): the name of the wav file.
        If not specified, the name will be the same as the source file.

    Returns:
        Path of the converted file.
    """
    source = Path(source)
    start = int(start * 1000)
    end = int(end * 1000) if end is not None else None
    wav_file = source.with_suffix(".wav")
    if wav_file == source:
        if _is_valid_wav_file(source):
            return source
        wav_file = Path(f'{source.with_suffix("")}_converted.wav')

    output_path = wav_file if output is None else Path(output)

    args = {"ss": start, "loglevel": "warning"}
    if end is not None:
        args["to"] = end

    # TODO: make a proper progress bar (then not hide with loglevel=warning)
    try:
        (
            ffmpeg.input(str(source), **args)
            .output(str(output_path), acodec="pcm_s16le", ac=1)
            .run()
        )
    except ffmpeg.Error as e:
        warnings.warn(f"Error during the conversion: {e})", stacklevel=1)
        return output_path
    return output_path


def parse_data_buffer(
    data: bytes, recognizer: vosk.KaldiRecognizer
) -> tuple[float, str] | None:
    """Parse the data buffer obtained from the recognizer.
    Return the time and the text.
    """
    if not recognizer.AcceptWaveform(data):
        msg = "No result found in the recognizer"
        warnings.warn(msg, stacklevel=2)
        return None

    result_dict = json.loads(recognizer.Result())
    try:
        time = result_dict["result"][0]["start"]
    except (IndexError, KeyError):
        warnings.warn("No time found in the result", stacklevel=2)
        return None
    return time, result_dict.get("text", "")


def transcribe(
    input_file: StrPath, model_path: StrPath, max_size: int | None = None
) -> Transcript:
    """Transcribe a  mono PCM 16-bit WAV file using a vosk model
    (https://alphacephei.com/vosk/models).

    Args:
        input_file (str): the transcript file, must be a mono PCM 16-bit WAV file.
        model_path (str): the vosk model path
        max_size (bool, optional): Transcribe only the first max_size bytes of the file.
        If not specified, the whole file will be transcribed.

    Raises:
        FileNotFoundError: A file is not found at the given path
        TypeError: The input file is not a wav file

    Returns:
        Transcript: the transcript of the file
    """
    input_file = Path(input_file)
    model_path = Path(model_path)

    if not input_file.is_file():
        msg = f"{input_file} not found"
        raise FileNotFoundError(msg)

    if not model_path.exists():
        msg = f"{model_path} not found"
        raise FileNotFoundError(msg)

    if not _is_valid_wav_file(input_file):
        msg = f"{input_file} is not a valid WAV file"
        raise TypeError(msg)
    model = vosk.Model(str(model_path))
    rec = _initialize_recognizer(model, input_file)

    return transcribe_with_vosk(input_file, rec, max_size)


def _is_valid_wav_file(input_file: Path) -> bool:
    """Validate if the input file is a valid WAV file."""
    try:
        wf = wave.Wave_read(str(input_file))
    except wave.Error as e:
        # if it is not a valid wav file for wave_read itself
        if "unknown format" in str(e):
            return False
        raise e from None

    is_mono = wf.getnchannels() == 1
    is_pcm = wf.getcomptype() == "NONE"
    is_16bit = wf.getsampwidth() == 2  # noqa: PLR2004
    return is_mono and is_pcm and is_16bit


def _initialize_recognizer(model: vosk.Model, input_file: Path) -> vosk.KaldiRecognizer:
    """Initialize the Vosk recognizer."""
    wave_form = wave.Wave_read(str(input_file))
    rec = vosk.KaldiRecognizer(model, wave_form.getframerate())

    # enable_words=True to get the time of each word
    # and in particular, the total time of the line.
    # otherwise, no time will be returned in rec.Result()
    # see the data_buffer function to see how we parse the result
    rec.SetWords(enable_words=True)
    return rec


def transcribe_with_vosk(
    input_file: StrPath, rec: vosk.KaldiRecognizer, max_size: int | None
) -> Transcript:
    """Transcribe the file using the Vosk recognizer."""
    input_file = Path(input_file)

    wave_form = wave.Wave_read(str(input_file))
    file_size = input_file.stat().st_size
    if max_size is not None and max_size < file_size:
        file_size = max_size
    pbar = tqdm(
        total=file_size, unit="B", unit_scale=True, desc=f"Transcribe {input_file}"
    )

    time_end = wave_form.getnframes() / wave_form.getframerate()
    transcript = Transcript(time_end=time_end)
    total_data = 0
    len_data = 1  # initialize with 1 to enter the loop
    while len_data > 0 and total_data < file_size:
        # use buffer of 4000 bytes
        data = wave_form.readframes(20000)
        len_data = len(data)
        total_data += len_data
        pbar.update(len_data)
        line_transcript = parse_data_buffer(data, rec)
        if line_transcript is not None:
            transcript.append(*line_transcript)

    return transcript


AllTranscriptFormats = TranscriptFormat | Literal["all"]


class ArgumentParser(tap.Tap):
    """Transcribe a file and optionally translate the transcript."""

    input: Path
    "the path to the audio file"

    output: Path | None = None
    """
    the path to the output file. Default: same as the input file with only the extension
    changed
    """

    format: str = "all"
    """
    the format of the transcript. Must be one of 'csv', 'json', 'srt', 'txt', 'vtt'
    or 'all'. Default: 'all'
    """

    model: Path = Path("model")
    "the path to the vosk model"

    lan_output: str | None = None  # the language of the transcript
    "the language of the output transcript. Default: no translation"

    lan_input: str = "auto"  # the language of the model
    "the language of the model. Default: auto"

    start: int = 0  # the time in seconds to start the conversion
    "the time in seconds to start the conversion"

    end: int | None = None  # the time in seconds to end the conversion
    "the time in seconds to end the conversion"

    max_size: int | None = None  # transcribe only the first max_size bytes of the file
    "transcribe only the first max_size bytes of the file"

    keep_wav: bool = False
    "keep the wav file after the transcript"

    verbosity: int = 2
    """Verbosity level. 0: no output, 1: only errors, 2: errors, info and progressbar,
    3: debug. Default: 2."""

    def process_args(self):
        if self.format not in typing.get_args(AllTranscriptFormats):
            msg = f"bad transcript format: {self.format}"
            raise ValueError(msg)

        vosk.SetLogLevel(-1)  # disable vosk logs
        match self.verbosity:
            case 0:
                logging.disable()
            case 1:
                logging.basicConfig(level=logging.ERROR)
            case 2:
                logging.basicConfig(level=logging.INFO)
            case 3:
                logging.basicConfig(level=logging.DEBUG)
                vosk.SetLogLevel(0)
            case _:
                msg = "verbosity must be 0, 1, 2 or 3"
                raise ValueError(msg)

    def configure(self):
        self.add_argument("input")
        self.add_argument("-o", "--output")
        self.add_argument("-f", "--format")
        self.add_argument("-m", "--model")
        self.add_argument("-li", "--lan_input")
        self.add_argument("-lo", "--lan_output")
        self.add_argument("-s", "--start")
        self.add_argument("-e", "--end")
        self.add_argument("-v", "--verbosity")

    def get_output(self, fmt: TranscriptFormat) -> Path:
        if self.output is None:
            if self.format == "all":
                return self.input.with_suffix(f".{fmt}")
            return self.input.with_suffix(f".{fmt}")
        return self.output

    def translate(self, transcript: Transcript):
        if self.lan_output is None:
            return transcript

        new_transcript, errors = transcript.translate(self.lan_output)
        if errors:
            lines = (f"{time} : {line} : {error}" for time, line, error in errors)
            logging.warning(f"Errors during the translation: {"\n".join(lines)}")
        return new_transcript


# ruff: noqa: G004
def main():
    logging.basicConfig(level=logging.INFO)
    parser = ArgumentParser()
    args = parser.parse_args()
    logging.info(f"Convert {args.input} to WAV format")
    wav_file = to_valid_wav(args.input, start=args.start, end=args.end)

    logging.info(f"Transcribe {wav_file}...")
    transcript = transcribe(wav_file, args.model, args.max_size)

    if not args.keep_wav:
        wav_file.unlink()

    transcript.language = args.lan_input

    new_transcript = args.translate(transcript)

    if args.format == "all":
        for fmt in TRANSCRIPT_FORMATS:
            new_transcript.write(args.get_output(fmt), fmt)
    else:
        new_transcript.write(args.get_output(args.format), args.format)
