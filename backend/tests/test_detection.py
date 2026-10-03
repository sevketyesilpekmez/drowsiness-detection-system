from app import create_app
from app.core.states import DetectionState


def test_detection_status_endpoint_shape():
    app = create_app()
    client = app.test_client()

    response = client.get("/api/detection/status")
    data = response.get_json()

    assert response.status_code == 200
    assert "state" in data
    assert "is_drowsy" in data
    assert "confidence" in data
    assert "error" in data


def test_detection_status_reflects_shared_state_default_when_engine_not_started():
    """Testlerde ENABLE_DETECTION_ENGINE hiç set edilmez, dolayısıyla engine
    hiç başlamaz ve endpoint SharedState'in kurucudaki varsayılanını
    (STARTING) döner — eski sabit 'NOT_STARTED' stub'ı artık yok."""
    app = create_app()
    client = app.test_client()

    response = client.get("/api/detection/status")
    data = response.get_json()

    assert data["state"] == DetectionState.STARTING
    assert data["is_drowsy"] is False
    assert data["error"] is None


def test_detection_status_reflects_live_shared_state_updates():
    """Route'un gerçekten app.extensions['shared_state']'i okuduğunu
    kanıtlar: state'i elle güncelleyip endpoint'in bunu yansıttığını
    doğrular (DetectionEngine'i hiç devreye sokmadan)."""
    app = create_app()
    shared_state = app.extensions["shared_state"]

    shared_state.update(
        frame=b"fake-jpeg",
        result={
            "state": DetectionState.DROWSY,
            "is_drowsy": True,
            "eyes_closed": True,
            "confidence": 0.97,
            "closed_duration": 3.1,
            "fps": 24.0,
        },
    )

    client = app.test_client()
    response = client.get("/api/detection/status")
    data = response.get_json()

    assert data["state"] == DetectionState.DROWSY
    assert data["is_drowsy"] is True
    assert data["confidence"] == 0.97


def test_detection_status_includes_alarm_snapshot_available_flag():
    app = create_app()
    client = app.test_client()

    response = client.get("/api/detection/status")
    data = response.get_json()

    assert data["alarm_snapshot_available"] is False


# --- GET /api/detection/alarm-snapshot ---


def test_alarm_snapshot_returns_404_when_no_alarm_triggered_yet():
    app = create_app()
    client = app.test_client()

    response = client.get("/api/detection/alarm-snapshot")

    assert response.status_code == 404
    assert "error" in response.get_json()


def test_alarm_snapshot_returns_jpeg_after_alarm_triggered():
    """Route'un gerçekten app.extensions['shared_state']'i okuduğunu
    kanıtlar (bkz. test_detection_status_reflects_live_shared_state_updates)."""
    app = create_app()
    shared_state = app.extensions["shared_state"]

    shared_state.update(
        frame=b"\xff\xd8fake-alarm-jpeg",
        result={
            "state": DetectionState.DROWSY,
            "is_drowsy": True,
            "confidence": 0.93,
            "closed_duration": 2.1,
            "alarm_triggered": True,
        },
    )

    client = app.test_client()
    response = client.get("/api/detection/alarm-snapshot")

    assert response.status_code == 200
    assert response.mimetype == "image/jpeg"
    assert response.data == b"\xff\xd8fake-alarm-jpeg"


def test_detection_status_reflects_alarm_snapshot_available_after_trigger():
    app = create_app()
    shared_state = app.extensions["shared_state"]

    shared_state.update(
        frame=b"\xff\xd8fake-alarm-jpeg",
        result={"state": DetectionState.DROWSY, "is_drowsy": True, "alarm_triggered": True},
    )

    client = app.test_client()
    data = client.get("/api/detection/status").get_json()

    assert data["alarm_snapshot_available"] is True


def test_detection_status_includes_alarm_snapshot_closed_duration():
    """Mentor'ün isteği: görüntüyle birlikte 'ne kadar süre kapalı kaldığı'
    bilgisi de dönmeli — ayrı bir isteğe gerek kalmadan /api/detection/status
    üzerinden gelir."""
    app = create_app()
    shared_state = app.extensions["shared_state"]

    shared_state.update(
        frame=b"\xff\xd8fake-alarm-jpeg",
        result={
            "state": DetectionState.DROWSY,
            "is_drowsy": True,
            "confidence": 0.88,
            "closed_duration": 2.7,
            "alarm_triggered": True,
        },
    )

    client = app.test_client()
    data = client.get("/api/detection/status").get_json()

    assert data["alarm_snapshot_closed_duration"] == 2.7
    assert data["alarm_snapshot_confidence"] == 0.88
    assert data["alarm_snapshot_state"] == DetectionState.DROWSY


# --- GET /api/detection/alarm-history ---


def test_alarm_history_is_empty_list_before_any_alarm():
    app = create_app()
    client = app.test_client()

    response = client.get("/api/detection/alarm-history")
    data = response.get_json()

    assert response.status_code == 200
    assert data["history"] == []


def test_alarm_history_reflects_triggered_alarms_newest_first():
    app = create_app()
    shared_state = app.extensions["shared_state"]

    for i in range(3):
        shared_state.update(
            frame=f"frame-{i}".encode(),
            result={"state": DetectionState.DROWSY, "is_drowsy": True, "alarm_triggered": True},
        )

    client = app.test_client()
    data = client.get("/api/detection/alarm-history").get_json()

    assert [entry["id"] for entry in data["history"]] == [3, 2, 1]
    assert "frame" not in data["history"][0]


# --- GET /api/detection/alarm-snapshot/<id> ---


def test_alarm_snapshot_by_id_returns_404_for_unknown_id():
    app = create_app()
    client = app.test_client()

    response = client.get("/api/detection/alarm-snapshot/999")

    assert response.status_code == 404
    assert "error" in response.get_json()


def test_alarm_snapshot_by_id_returns_correct_jpeg():
    app = create_app()
    shared_state = app.extensions["shared_state"]

    shared_state.update(
        frame=b"\xff\xd8first",
        result={"state": DetectionState.DROWSY, "is_drowsy": True, "alarm_triggered": True},
    )
    shared_state.update(
        frame=b"\xff\xd8second",
        result={"state": DetectionState.DROWSY, "is_drowsy": True, "alarm_triggered": True},
    )

    client = app.test_client()

    first_response = client.get("/api/detection/alarm-snapshot/1")
    second_response = client.get("/api/detection/alarm-snapshot/2")

    assert first_response.status_code == 200
    assert first_response.mimetype == "image/jpeg"
    assert first_response.data == b"\xff\xd8first"

    assert second_response.status_code == 200
    assert second_response.data == b"\xff\xd8second"
