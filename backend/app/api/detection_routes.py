from flask import Blueprint, Response, jsonify

from app.extensions import get_shared_state

detection_bp = Blueprint("detection", __name__)


@detection_bp.get("/detection/status")
def get_detection_status():
    """
    Son uyku tespiti sonucunu SharedState'ten okuyup JSON olarak döndürür.
    Detection engine henüz başlamadıysa (ör. ENABLE_DETECTION_ENGINE=false)
    SharedState'in kurucudaki varsayılan durumu döner: state=STARTING.
    """
    shared_state = get_shared_state()
    return jsonify(shared_state.get_result())


@detection_bp.get("/detection/alarm-snapshot")
def get_alarm_snapshot():
    """
    En son tetiklenen alarmın anındaki kareyi ham JPEG olarak döndürür —
    yalnızca alarmın çaldığını değil, sebep olan görüntüyü de gösterebilmek
    için (bkz. SharedState.get_alarm_snapshot).

    Frontend bunu <img src="/api/detection/alarm-snapshot?..."> ile
    doğrudan kullanır; /api/detection/status'teki alarm_snapshot_available
    alanı bu endpoint'e gitmeden "gösterecek bir şey var mı" diye sorar.

    Henüz hiç alarm tetiklenmediyse 404 + JSON hata döner.
    """
    shared_state = get_shared_state()
    snapshot = shared_state.get_alarm_snapshot()

    if snapshot is None or snapshot.get("frame") is None:
        return jsonify({"error": "Henüz tetiklenmiş bir alarm yok."}), 404

    return Response(snapshot["frame"], mimetype="image/jpeg")


@detection_bp.get("/detection/alarm-history")
def get_alarm_history():
    """
    Son ALARM_HISTORY_MAX_LEN (5) alarmın metadata listesini döndürür (en
    yeni önce). Görüntü burada yok (ağır olduğu için); her kaydın görüntüsü
    ayrıca GET /detection/alarm-snapshot/<id>'den çekilir.
    """
    shared_state = get_shared_state()
    return jsonify({"history": shared_state.get_alarm_history()})


@detection_bp.get("/detection/alarm-snapshot/<int:alarm_id>")
def get_alarm_snapshot_by_id(alarm_id: int):
    """Alarm geçmişindeki belirli bir kaydın JPEG görüntüsünü döndürür.
    O id artık geçmişte yoksa (ör. ALARM_HISTORY_MAX_LEN'i aşıp düştüyse)
    veya hiç var olmadıysa 404 döner."""
    shared_state = get_shared_state()
    snapshot = shared_state.get_alarm_snapshot_by_id(alarm_id)

    if snapshot is None or snapshot.get("frame") is None:
        return jsonify({"error": "Böyle bir alarm kaydı yok."}), 404

    return Response(snapshot["frame"], mimetype="image/jpeg")
