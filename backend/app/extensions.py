from __future__ import annotations

from flask import Flask, current_app

from app.core.shared_state import SharedState


def get_shared_state(app: Flask | None = None) -> SharedState:
    """
    current_app.extensions içinden SharedState'i tip-güvenli şekilde
    döndürür. detection_routes.py ve stream_routes.py'nin ikisi de bunu
    kullanır — aynı `_get_shared_state()` mantığını iki dosyada ayrı ayrı
    tanımlamak yerine tek yerde toplanır.

    SharedState app.extensions'a bağlanmamışsa (create_app() içindeki
    wiring atlanmış/bozulmuşsa) RuntimeError fırlatır — bu durumda sahte
    veri dönmek yerine gürültülü şekilde patlamak tercih edilir, çünkü bu
    bir konfigürasyon hatasıdır.
    """
    target_app = app or current_app
    shared_state = target_app.extensions.get("shared_state")

    if not isinstance(shared_state, SharedState):
        raise RuntimeError(
            "SharedState Flask uygulamasına bağlanmamış "
            "(app.extensions['shared_state'] eksik). create_app() içindeki "
            "wiring'i kontrol edin."
        )

    return shared_state
