from app import create_app
from app.api.stream_routes import _generate_mjpeg_stream, _should_yield
from app.core.shared_state import SharedState
from app.core.states import DetectionState


def _dummy_result(**overrides):
    result = {
        "state": DetectionState.AWAKE,
        "is_drowsy": False,
        "eyes_closed": False,
        "confidence": 0.5,
        "closed_duration": 0.0,
    }
    result.update(overrides)
    return result


# --- _should_yield: saf karar mantığı, thread/generator'dan bağımsız ---
# Not: generator'ı birden fazla thread'den aynı anda sürmek (next() +
# close()'u eşzamanlı çağırmak) Python'da güvenli değildir (generator
# "already executing" durumuna girer). Bu yüzden "bekliyor mu" davranışını
# generator üzerinden thread'lerle kanıtlamak yerine, karar mantığını ayrı
# bir saf fonksiyona çıkarıp onu doğrudan test ediyoruz.


def test_should_yield_false_when_no_frame_yet():
    assert _should_yield(frame=None, frame_id=0, last_frame_id=-1) is False


def test_should_yield_false_when_frame_id_unchanged():
    assert _should_yield(frame=b"x", frame_id=5, last_frame_id=5) is False


def test_should_yield_true_when_frame_id_changed():
    assert _should_yield(frame=b"x", frame_id=6, last_frame_id=5) is True


def test_should_yield_true_on_first_ever_frame():
    assert _should_yield(frame=b"x", frame_id=0, last_frame_id=-1) is True


# --- _generate_mjpeg_stream: tek next() çağrısıyla format doğrulaması ---
# Generator sonsuz olduğu için tek bir next() çağırıp hemen kapatıyoruz;
# bu güvenlidir çünkü generator, close() çağrıldığında zaten bir yield
# noktasında askıda bekliyor (başka bir thread onu aynı anda sürmüyor).


def test_generate_mjpeg_stream_yields_formatted_chunk():
    shared_state = SharedState()
    shared_state.update(frame=b"\xff\xd8FRAME1", result=_dummy_result())

    generator = _generate_mjpeg_stream(shared_state)
    try:
        chunk = next(generator)
        assert chunk.startswith(b"--frame\r\n")
        assert b"Content-Type: image/jpeg" in chunk
        assert b"FRAME1" in chunk
    finally:
        generator.close()


def test_generate_mjpeg_stream_yields_second_frame_after_update():
    shared_state = SharedState()
    shared_state.update(frame=b"\xff\xd8FRAME1", result=_dummy_result())

    generator = _generate_mjpeg_stream(shared_state)
    try:
        first_chunk = next(generator)
        assert b"FRAME1" in first_chunk

        shared_state.update(frame=b"\xff\xd8FRAME2", result=_dummy_result())

        second_chunk = next(generator)
        assert b"FRAME2" in second_chunk
    finally:
        generator.close()


def test_stream_route_returns_correct_mimetype():
    """
    ÖNEMLİ: client.get("/api/stream") KULLANMIYORUZ. Flask/Werkzeug'un test
    client'ı (bu kurulu sürümde) generator tabanlı streaming response'ları
    varsayılan olarak İSTEKLİYCE (eagerly) tüketmeye çalışıyor — sonsuz bir
    generator için bu sonsuza kadar bloklar (bu test dosyasının ilk
    versiyonunda tam olarak bu oldu, tüm test suite'ini kilitledi).

    Bunun yerine view fonksiyonunu test_request_context içinde DOĞRUDAN
    çağırıyoruz; bu, tam HTTP/WSGI dispatch'ini (ve dolayısıyla eager
    consumption'ı) atlar. Generator hiç next() edilmediği için askıda
    kalır, close() güvenlidir.
    """
    from app.api.stream_routes import video_stream

    app = create_app()
    with app.test_request_context("/api/stream"):
        response = video_stream()
        try:
            assert response.status_code == 200
            assert response.mimetype == "multipart/x-mixed-replace"
        finally:
            response.response.close()
