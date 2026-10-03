import threading

from app.core.shared_state import SharedState
from app.core.states import DetectionState


def test_initial_state_is_starting():
    state = SharedState()
    result = state.get_result()

    assert result["state"] == DetectionState.STARTING
    assert result["is_drowsy"] is False
    assert result["error"] is None
    assert state.get_frame() is None


def test_update_sets_frame_and_result():
    state = SharedState()

    state.update(
        frame=b"fake-jpeg-frame",
        result={
            "state": DetectionState.AWAKE,
            "is_drowsy": False,
            "eyes_closed": False,
            "confidence": 0.95,
            "closed_duration": 0.0,
            "fps": 30.0,
        },
    )

    result = state.get_result()
    assert result["state"] == DetectionState.AWAKE
    assert result["confidence"] == 0.95
    assert result["error"] is None
    assert state.get_frame() == b"fake-jpeg-frame"


def test_update_adds_timestamp_automatically():
    state = SharedState()
    state.update(frame=b"x", result={"state": DetectionState.AWAKE, "is_drowsy": False})

    assert "timestamp" in state.get_result()


def test_update_result_does_not_touch_frame():
    state = SharedState()
    state.update(frame=b"original-frame", result={"state": DetectionState.AWAKE, "is_drowsy": False})

    state.update_result(result={"state": DetectionState.DROWSY, "is_drowsy": True})

    assert state.get_frame() == b"original-frame"
    assert state.get_result()["state"] == DetectionState.DROWSY


def test_set_error_resets_to_safe_defaults():
    state = SharedState()
    state.update(
        frame=b"x",
        result={
            "state": DetectionState.AWAKE,
            "is_drowsy": True,
            "confidence": 0.99,
        },
    )

    state.set_error("Kamera açılamadı.", state=DetectionState.CAMERA_ERROR)
    result = state.get_result()

    assert result["state"] == DetectionState.CAMERA_ERROR
    assert result["is_drowsy"] is False
    assert result["confidence"] == 0.0
    assert result["error"] == "Kamera açılamadı."


def test_set_error_default_state_is_system_error():
    state = SharedState()
    state.set_error("bilinmeyen hata")

    assert state.get_result()["state"] == DetectionState.SYSTEM_ERROR


def test_clear_error_removes_error_but_keeps_state():
    state = SharedState()
    state.set_error("gecici hata", state=DetectionState.CAMERA_ERROR)
    state.clear_error()

    result = state.get_result()
    assert result["error"] is None
    assert result["state"] == DetectionState.CAMERA_ERROR


def test_get_snapshot_returns_consistent_triplet():
    state = SharedState()
    state.update(frame=b"frame-1", result={"state": DetectionState.AWAKE, "is_drowsy": False})

    snapshot = state.get_snapshot()
    assert snapshot["frame"] == b"frame-1"
    assert snapshot["result"]["state"] == DetectionState.AWAKE
    assert snapshot["error"] is None


def test_get_result_returns_independent_copy():
    """dict(...) shallow copy'nin caller tarafındaki mutasyona karşı
    yeterli korumayı sağladığını doğrular (deepcopy'ye gerek yok)."""
    state = SharedState()
    state.update(frame=b"x", result={"state": DetectionState.AWAKE, "is_drowsy": False})

    result = state.get_result()
    result["state"] = "TAMPERED"

    assert state.get_result()["state"] == DetectionState.AWAKE


def test_concurrent_updates_do_not_raise():
    """Bu sınıf projenin omurgası; kilidin gerçekten çalıştığını görmeden
    güvenmiyoruz. Birden fazla writer + reader thread aynı anda çalışır."""
    state = SharedState()
    errors: list[Exception] = []

    def writer():
        for i in range(200):
            state.update(
                frame=f"frame-{i}".encode(),
                result={
                    "state": DetectionState.AWAKE,
                    "is_drowsy": False,
                    "confidence": i / 200,
                },
            )

    def reader():
        for _ in range(200):
            try:
                state.get_result()
                state.get_frame()
                state.get_snapshot()
            except Exception as exc:  # pragma: no cover
                errors.append(exc)

    threads = [threading.Thread(target=writer) for _ in range(3)]
    threads += [threading.Thread(target=reader) for _ in range(3)]

    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert errors == []


def test_frame_id_starts_at_zero():
    state = SharedState()
    frame_id, frame = state.get_frame_with_id()

    assert frame_id == 0
    assert frame is None


def test_frame_id_increments_on_update():
    state = SharedState()

    state.update(frame=b"frame-1", result={"state": DetectionState.AWAKE, "is_drowsy": False})
    first_id, first_frame = state.get_frame_with_id()

    state.update(frame=b"frame-2", result={"state": DetectionState.AWAKE, "is_drowsy": False})
    second_id, second_frame = state.get_frame_with_id()

    assert second_id == first_id + 1
    assert first_frame == b"frame-1"
    assert second_frame == b"frame-2"


def test_frame_id_does_not_increment_on_update_result():
    """update_result() frame'e dokunmuyor; frame_id de değişmemeli —
    stream endpoint'i bunu 'yeni frame yok' olarak doğru yorumlamalı."""
    state = SharedState()
    state.update(frame=b"frame-1", result={"state": DetectionState.AWAKE, "is_drowsy": False})
    id_after_update, _ = state.get_frame_with_id()

    state.update_result(result={"state": DetectionState.DROWSY, "is_drowsy": True})
    id_after_update_result, _ = state.get_frame_with_id()

    assert id_after_update_result == id_after_update


def test_frame_id_does_not_increment_on_set_error():
    state = SharedState()
    state.update(frame=b"frame-1", result={"state": DetectionState.AWAKE, "is_drowsy": False})
    id_before_error, _ = state.get_frame_with_id()

    state.set_error("kamera hatası", state=DetectionState.CAMERA_ERROR)
    id_after_error, _ = state.get_frame_with_id()

    assert id_after_error == id_before_error


def test_get_snapshot_includes_frame_id():
    state = SharedState()
    state.update(frame=b"frame-1", result={"state": DetectionState.AWAKE, "is_drowsy": False})

    snapshot = state.get_snapshot()

    assert snapshot["frame_id"] == 1


# --- alarm anı görüntüsü (get_alarm_snapshot) ---
# Mentor'ün istediği şey: sadece alarmın çaldığını değil, alarmın hangi
# görüntüye bakılarak verildiğini de görebilmek. alarm_triggered=True olan
# bir update() çağrısı, o kareyi ayrıca "alarm görüntüsü" olarak saklamalı.


def test_get_alarm_snapshot_is_none_before_any_alarm():
    state = SharedState()
    state.update(frame=b"frame-1", result={"state": DetectionState.AWAKE, "is_drowsy": False})

    assert state.get_alarm_snapshot() is None


def test_normal_update_without_alarm_triggered_does_not_set_snapshot():
    state = SharedState()
    state.update(
        frame=b"frame-1",
        result={
            "state": DetectionState.DROWSY,
            "is_drowsy": True,
            "alarm_active": True,
            "alarm_triggered": False,
        },
    )

    assert state.get_alarm_snapshot() is None


def test_alarm_triggered_update_captures_frame_and_context():
    state = SharedState()
    state.update(
        frame=b"alarm-frame",
        result={
            "state": DetectionState.DROWSY,
            "is_drowsy": True,
            "confidence": 0.91,
            "closed_duration": 2.4,
            "alarm_triggered": True,
        },
    )

    snapshot = state.get_alarm_snapshot()

    assert snapshot["frame"] == b"alarm-frame"
    assert snapshot["state"] == DetectionState.DROWSY
    assert snapshot["confidence"] == 0.91
    assert snapshot["closed_duration"] == 2.4
    assert "timestamp" in snapshot


def test_alarm_triggered_without_frame_does_not_crash_or_set_snapshot():
    """alarm_triggered=True gelse bile frame None ise (ör. encode hatası
    sonrası bir yol) çökmemeli ve önceki durumu bozmamalı."""
    state = SharedState()
    state.update(frame=None, result={"state": DetectionState.DROWSY, "alarm_triggered": True})

    assert state.get_alarm_snapshot() is None


def test_new_alarm_overwrites_previous_snapshot():
    state = SharedState()
    state.update(frame=b"first-alarm", result={"state": DetectionState.DROWSY, "alarm_triggered": True})
    state.update(frame=b"second-alarm", result={"state": DetectionState.DROWSY, "alarm_triggered": True})

    assert state.get_alarm_snapshot()["frame"] == b"second-alarm"


def test_non_triggering_update_after_alarm_keeps_previous_snapshot():
    """Alarm bir kez tetiklendikten sonra, alarm_triggered=False olan
    sonraki normal karelerin snapshot'ı SİLMEMESİ gerekir — mentor'ün
    görmek istediği 'alarmın verildiği görüntü' sabit kalmalı."""
    state = SharedState()
    state.update(frame=b"alarm-frame", result={"state": DetectionState.DROWSY, "alarm_triggered": True})
    state.update(frame=b"later-frame", result={"state": DetectionState.AWAKE, "alarm_triggered": False})

    assert state.get_alarm_snapshot()["frame"] == b"alarm-frame"


def test_get_alarm_snapshot_returns_independent_copy():
    state = SharedState()
    state.update(frame=b"alarm-frame", result={"state": DetectionState.DROWSY, "alarm_triggered": True})

    snapshot = state.get_alarm_snapshot()
    snapshot["state"] = "TAMPERED"

    assert state.get_alarm_snapshot()["state"] == DetectionState.DROWSY


def test_get_result_reports_alarm_snapshot_availability():
    state = SharedState()
    assert state.get_result()["alarm_snapshot_available"] is False

    state.update(frame=b"alarm-frame", result={"state": DetectionState.DROWSY, "alarm_triggered": True})

    assert state.get_result()["alarm_snapshot_available"] is True


def test_get_result_includes_alarm_snapshot_context_fields_before_any_alarm():
    """Anahtarlar alarm hiç tetiklenmeden önce de her zaman mevcut olmalı
    (None değerleriyle) — sabit bir sözleşme, frontend'in undefined kontrolü
    yapmasına gerek kalmasın."""
    state = SharedState()
    result = state.get_result()

    assert result["alarm_snapshot_closed_duration"] is None
    assert result["alarm_snapshot_confidence"] is None
    assert result["alarm_snapshot_state"] is None
    assert result["alarm_snapshot_timestamp"] is None


def test_get_result_includes_alarm_snapshot_context_fields_after_alarm():
    """Mentor'ün isteği: görüntüyle birlikte 'ne kadar süre kapalı kaldığı'
    gibi bağlam bilgisi de gösterilmeli — bu, ayrı bir istek gerektirmeden
    /api/detection/status'un kendisinden gelir."""
    state = SharedState()
    state.update(
        frame=b"alarm-frame",
        result={
            "state": DetectionState.DROWSY,
            "confidence": 0.91,
            "closed_duration": 2.6,
            "alarm_triggered": True,
        },
    )

    result = state.get_result()

    assert result["alarm_snapshot_closed_duration"] == 2.6
    assert result["alarm_snapshot_confidence"] == 0.91
    assert result["alarm_snapshot_state"] == DetectionState.DROWSY
    assert "timestamp" in result and result["alarm_snapshot_timestamp"] is not None


def test_get_result_includes_alarm_snapshot_id():
    state = SharedState()
    assert state.get_result()["alarm_snapshot_id"] is None

    state.update(frame=b"alarm-frame", result={"state": DetectionState.DROWSY, "alarm_triggered": True})

    assert state.get_result()["alarm_snapshot_id"] == 1


# --- alarm geçmişi (get_alarm_history / get_alarm_snapshot_by_id) ---
# Mentor notu: "Son 3-5 alarm yeterli" — SharedState en fazla
# ALARM_HISTORY_MAX_LEN (5) alarmı görüntüsüyle birlikte bellekte tutar.


def test_get_alarm_history_is_empty_before_any_alarm():
    state = SharedState()

    assert state.get_alarm_history() == []


def test_get_alarm_history_excludes_frame_bytes():
    """Geçmiş listesi metadata-only olmalı (ağır JPEG'ler burada olmamalı) —
    her kaydın görüntüsü ayrıca get_alarm_snapshot_by_id(id) ile çekilir."""
    state = SharedState()
    state.update(
        frame=b"alarm-frame",
        result={"state": DetectionState.DROWSY, "confidence": 0.9, "closed_duration": 2.5, "alarm_triggered": True},
    )

    history = state.get_alarm_history()

    assert len(history) == 1
    entry = history[0]
    assert "frame" not in entry
    assert entry["id"] == 1
    assert entry["state"] == DetectionState.DROWSY
    assert entry["confidence"] == 0.9
    assert entry["closed_duration"] == 2.5
    assert "timestamp" in entry


def test_get_alarm_history_orders_newest_first():
    state = SharedState()
    for i in range(3):
        state.update(
            frame=f"frame-{i}".encode(),
            result={"state": DetectionState.DROWSY, "alarm_triggered": True},
        )

    history = state.get_alarm_history()

    assert [entry["id"] for entry in history] == [3, 2, 1]


def test_get_alarm_history_is_bounded_to_max_len():
    """8 alarm tetiklense bile yalnızca en yeni 5'i tutulmalı (bkz.
    ALARM_HISTORY_MAX_LEN) — uzun bir sürüşte sınırsız büyümeyi önler."""
    state = SharedState()
    for i in range(8):
        state.update(
            frame=f"frame-{i}".encode(),
            result={"state": DetectionState.DROWSY, "alarm_triggered": True},
        )

    history = state.get_alarm_history()

    assert len(history) == 5
    assert [entry["id"] for entry in history] == [8, 7, 6, 5, 4]


def test_get_alarm_history_ignores_non_triggering_updates():
    state = SharedState()
    state.update(frame=b"alarm-frame", result={"state": DetectionState.DROWSY, "alarm_triggered": True})
    state.update(frame=b"later-frame", result={"state": DetectionState.AWAKE, "alarm_triggered": False})

    assert len(state.get_alarm_history()) == 1


def test_get_alarm_snapshot_by_id_returns_matching_frame():
    state = SharedState()
    state.update(frame=b"first-alarm", result={"state": DetectionState.DROWSY, "alarm_triggered": True})
    state.update(frame=b"second-alarm", result={"state": DetectionState.DROWSY, "alarm_triggered": True})

    first = state.get_alarm_snapshot_by_id(1)
    second = state.get_alarm_snapshot_by_id(2)

    assert first["frame"] == b"first-alarm"
    assert second["frame"] == b"second-alarm"


def test_get_alarm_snapshot_by_id_returns_none_for_unknown_id():
    state = SharedState()
    state.update(frame=b"alarm-frame", result={"state": DetectionState.DROWSY, "alarm_triggered": True})

    assert state.get_alarm_snapshot_by_id(999) is None


def test_get_alarm_snapshot_by_id_returns_none_after_falling_out_of_history():
    """maxlen=5'i aşıp deque'den düşen bir id artık bulunamamalı."""
    state = SharedState()
    for i in range(6):
        state.update(
            frame=f"frame-{i}".encode(),
            result={"state": DetectionState.DROWSY, "alarm_triggered": True},
        )

    # id=1, 6. alarmda deque'den düşmüş olmalı (maxlen=5: [2,3,4,5,6] kalır).
    assert state.get_alarm_snapshot_by_id(1) is None
    assert state.get_alarm_snapshot_by_id(6) is not None
