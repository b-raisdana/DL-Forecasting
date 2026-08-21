import logging

import pytest
from loguru import logger

from app.helper.logging.do_log.log_it import _nearest_level_name, log


@pytest.mark.unit
class TestNearestLevelName:
    @pytest.mark.parametrize(
        "severity,expected",
        [
            (logging.DEBUG, "DEBUG"),
            (logging.INFO, "INFO"),
            (logging.WARNING, "WARNING"),
            (logging.ERROR, "ERROR"),
            (logging.CRITICAL, "CRITICAL"),
        ],
    )
    def test_standard_levels(self, severity, expected):
        assert _nearest_level_name(severity) == expected

    @pytest.mark.parametrize(
        "severity,expected",
        [
            (25, "INFO"),
            (35, "WARNING"),
            (45, "ERROR"),
            (55, "CRITICAL"),
            (5, "DEBUG"),
            (0, "DEBUG"),
            (-1, "DEBUG"),
        ],
    )
    def test_non_standard_levels(self, severity, expected):
        assert _nearest_level_name(severity) == expected


def _level_2(stack_limit, stack_offset=0):
    _level_3(stack_limit, stack_offset)


def _level_3(stack_limit, stack_offset):
    log("boom", logging.ERROR, stack_limit, stack_offset)


@pytest.mark.unit
class TestLogStackTrace:
    """Characterizes log()'s stack_limit/stack_offset frame-trimming so an optimization to how
    the trace is built (e.g. bounding traceback.format_stack's work) can be checked against the
    exact same frame count and identity, not just "doesn't crash"."""

    def _captured_message(self, **kwargs) -> str:
        sink: list[str] = []
        sink_id = logger.add(sink.append, format="{message}")
        try:
            _level_2(**kwargs)
        finally:
            logger.remove(sink_id)
        return sink[0]

    def test_stack_limit_zero_omits_trace(self):
        message = self._captured_message(stack_limit=0)
        assert 'File "' not in message

    def test_stack_limit_keeps_last_n_frames_nearest_call_site(self):
        message = self._captured_message(stack_limit=2)
        assert message.count('File "') == 2
        assert "in _level_2" in message
        assert "in _level_3" in message

    def test_stack_offset_skips_additional_frames_from_call_site(self):
        message = self._captured_message(stack_limit=1, stack_offset=1)
        assert message.count('File "') == 1
        assert "in _level_2" in message
        assert "in _level_3" not in message
