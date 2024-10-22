import re

import hypothesis.strategies as st
import pytest
from hypothesis import given

import pytranscript as pt


def seconds_strat(start, end):
    return st.floats(start, end, allow_nan=False, allow_infinity=False)


@given(seconds_strat(pt.SECONDS_IN_DAY, 2**32))
def test_seconds_to_time_value_day(seconds):
    formated_time = pt.seconds_to_time(seconds)
    match = re.match(r"^(\d+)d (\d{2}):(\d{2}):(\d{2})$", formated_time)
    assert match is not None
    result_time = (
        int(match[1]) * pt.SECONDS_IN_DAY
        + int(match[2]) * pt.SECONDS_IN_HOUR
        + int(match[3]) * pt.SECONDS_IN_MINUTE
        + int(match[4])
    )
    assert result_time == round(seconds)


@given(seconds_strat(pt.SECONDS_IN_HOUR, pt.SECONDS_IN_DAY - 1))
def test_seconds_to_time_value_hour(seconds):
    formated_time = pt.seconds_to_time(seconds)
    match = re.match(r"^(\d{2}):(\d{2}):(\d{2})$", formated_time)
    assert match is not None
    result_time = (
        int(match[1]) * pt.SECONDS_IN_HOUR
        + int(match[2]) * pt.SECONDS_IN_MINUTE
        + int(match[3])
    )
    assert result_time == round(seconds)


@given(seconds_strat(0, pt.SECONDS_IN_HOUR - 1))
def test_seconds_to_time_value_minute(seconds):
    formated_time = pt.seconds_to_time(seconds)
    match = re.match(r"^(\d{2}):(\d{2}\.\d{2})$", formated_time)
    assert match is not None
    result_time = int(match[1]) * pt.SECONDS_IN_MINUTE + float(match[2])
    # need approx because of the conversion str -> float
    assert pytest.approx(result_time) == round(seconds, 2)


def test_seconds_to_time_exception():
    with pytest.raises(ValueError, match=r".*seconds >= 2\*\*32 are not supported.*"):
        pt.seconds_to_time(2**32 + 1)


@given(seconds_strat(0, 2**32))
def test_seconds_to_srt(seconds):
    formated_time = pt.seconds_to_srt_time(seconds)
    match = re.match(r"^(\d+):(\d{2}):(\d{2}),(\d{3})$", formated_time)
    assert match is not None
    result_time = (
        int(match[1]) * pt.SECONDS_IN_HOUR
        + int(match[2]) * pt.SECONDS_IN_MINUTE
        + int(match[3])
        + int(match[4]) / 1000
    )
    assert result_time == round(seconds, 3)
