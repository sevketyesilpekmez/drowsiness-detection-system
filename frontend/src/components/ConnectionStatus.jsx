import { useEffect, useState } from "react";
import { getHealth } from "../services/api";

export default function ConnectionStatus() {
  const [connected, setConnected] = useState(null);

  useEffect(() => {
    let cancelled = false;

    async function check() {
      try {
        await getHealth();
        if (!cancelled) setConnected(true);
      } catch {
        if (!cancelled) setConnected(false);
      }
    }

    check();
    const interval = setInterval(check, 3000);

    return () => {
      cancelled = true;
      clearInterval(interval);
    };
  }, []);

  const label =
    connected === null ? "Kontrol ediliyor..." : connected ? "Backend bağlı" : "Backend'e ulaşılamıyor";

  const dotClass = connected === null ? "dot dot-pending" : connected ? "dot dot-ok" : "dot dot-error";

  return (
    <div className="connection-status">
      <span className={dotClass} />
      <span>{label}</span>
    </div>
  );
}
