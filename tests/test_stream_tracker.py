import asyncio

import pytest

from app.limits import StreamTracker


async def test_stream_tracker_waits_and_closes_admission():
    tracker = StreamTracker()
    await tracker.start()
    closing = asyncio.create_task(tracker.close(timeout=1))
    await asyncio.sleep(0)
    assert not closing.done()
    await tracker.finish()
    assert await closing is True
    with pytest.raises(Exception) as captured:
        await tracker.start()
    assert captured.value.status_code == 503


async def test_stream_tracker_reports_grace_timeout():
    tracker = StreamTracker()
    await tracker.start()
    assert await tracker.close(timeout=.001) is False
    await tracker.finish()
