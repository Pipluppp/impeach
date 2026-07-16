from __future__ import annotations

import pytest

from pipeline.transcribe_session import plan_chunks


def test_chunk_plan_has_overlap_but_contiguous_ownership() -> None:
    plans = plan_chunks(duration=3700, chunk_duration=1800, overlap=15)
    assert [(item.core_start, item.core_end) for item in plans] == [
        (0, 1800), (1800, 3600), (3600, 3700)
    ]
    assert [(item.input_start, item.input_end) for item in plans] == [
        (0, 1815), (1785, 3615), (3585, 3700)
    ]


@pytest.mark.parametrize(
    "duration,chunk,overlap",
    [(0, 10, 1), (10, 0, 1), (10, 10, -1), (10, 10, 5)],
)
def test_invalid_chunk_plan_fails(duration: float, chunk: float, overlap: float) -> None:
    with pytest.raises(ValueError):
        plan_chunks(duration, chunk, overlap)
