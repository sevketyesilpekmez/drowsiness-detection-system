import AlarmBanner from "../components/AlarmBanner";
import AlarmController from "../components/AlarmController";
import AlarmFlashOverlay from "../components/AlarmFlashOverlay";
import AlarmHistory from "../components/AlarmHistory";
import AlarmSnapshot from "../components/AlarmSnapshot";
import CameraFeed from "../components/CameraFeed";
import ConnectionStatus from "../components/ConnectionStatus";
import StatusCard from "../components/StatusCard";
import { useDetectionStatus } from "../hooks/useDetectionStatus";

export default function Dashboard() {
  const { status, connectionError, isLoading } = useDetectionStatus();
  const isAlarmActive = Boolean(status.alarm_active);

  return (
    <div className={isAlarmActive ? "dashboard alarm-mode" : "dashboard"}>
      <AlarmFlashOverlay active={isAlarmActive} />

      <header className="dashboard-header">
        <h1>Uyku Tespit Sistemi</h1>
        <ConnectionStatus />
      </header>

      <AlarmBanner status={status} />

      <main className="dashboard-main">
        <CameraFeed status={status} />
        <StatusCard status={status} connectionError={connectionError} isLoading={isLoading} />
      </main>

      <AlarmController status={status} />
      <AlarmSnapshot status={status} />
      <AlarmHistory alarmTriggerCount={status.alarm_trigger_count} />
    </div>
  );
}
