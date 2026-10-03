from __future__ import annotations

import time
from collections.abc import Iterator

from flask import Blueprint, Response

from app.core.shared_state import SharedState
from app.extensions import get_shared_state

stream_bp = Blueprint("stream", __name__)

# SharedState düz bir threading.Lock kullanıyor (Condition/notify değil) —
# birkaç izleyici için kısa aralıklı polling'in maliyeti önemsiz.
_POLL_INTERVAL_SECONDS = 0.03


def _should_yield(frame: bytes | None, frame_id: int, last_frame_id: int) -> bool:
    """Karar mantığı generator'dan ayrı, saf bir fonksiyon — threading/
    generator lifecycle'ıyla uğraşmadan test edilebilir. "Yeni frame var
    mı" sorusu frame_id ile cevaplanır (bytes identity yerine)."""
    return frame is not None and frame_id != last_frame_id


def _generate_mjpeg_stream(shared_state: SharedState) -> Iterator[bytes]:
    """SharedState içindeki son JPEG frame'i MJPEG formatında üretir."""
    last_frame_id = -1

    while True:
        frame_id, frame = shared_state.get_frame_with_id()

        if not _should_yield(frame, frame_id, last_frame_id):
            time.sleep(_POLL_INTERVAL_SECONDS)
            continue

        last_frame_id = frame_id

        yield (
            b"--frame\r\n"
            b"Content-Type: image/jpeg\r\n\r\n" + frame + b"\r\n"
        )


@stream_bp.get("/stream")
def video_stream() -> Response:
    """Canlı kamera görüntüsünü MJPEG stream olarak döndürür."""
    shared_state = get_shared_state()

    return Response(
        _generate_mjpeg_stream(shared_state),
        mimetype="multipart/x-mixed-replace; boundary=frame",
    )
