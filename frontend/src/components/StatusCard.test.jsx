import { render, screen } from "@testing-library/react";
import { describe, expect, it } from "vitest";
import StatusCard from "./StatusCard";
import { STATE_LABELS } from "../constants/detectionStates";

// Ürün kodunun sözlüğü doğrudan kullanılıyor — test kendi kopyasını
// tutsaydı, etiket değişince test değil sadece kopya güncellenir ve
// doğrulama anlamını yitirirdi.
describe("StatusCard", () => {
  it.each(Object.entries(STATE_LABELS))("%s state'ini doğru Türkçe etikete çevirir", (state, label) => {
    render(<StatusCard status={{ state }} connectionError={null} isLoading={false} />);

    expect(screen.getByText(label)).toBeInTheDocument();
  });

  it("bilinmeyen bir state için ham değeri gösterir (çökmez)", () => {
    render(<StatusCard status={{ state: "GELECEKTEKI_BILINMEYEN_STATE" }} connectionError={null} isLoading={false} />);

    expect(screen.getByText("GELECEKTEKI_BILINMEYEN_STATE")).toBeInTheDocument();
  });

  it("isLoading true iken 'Yükleniyor' gösterir", () => {
    render(<StatusCard status={{ state: "AWAKE" }} connectionError={null} isLoading={true} />);

    expect(screen.getByText("Yükleniyor")).toBeInTheDocument();
  });

  it("connectionError varsa hata mesajını gösterir", () => {
    render(<StatusCard status={{}} connectionError="Backend'e ulaşılamadı" isLoading={false} />);

    expect(screen.getByText("Backend'e ulaşılamadı")).toBeInTheDocument();
  });

  it("backend'in kendi error alanını da gösterir", () => {
    render(<StatusCard status={{ state: "CAMERA_ERROR", error: "Kamera açılamadı" }} connectionError={null} isLoading={false} />);

    expect(screen.getByText("Kamera açılamadı")).toBeInTheDocument();
  });

  it("AWAKE + gözler kapalı + süre birikiyorsa 'yaklaşıyor' modifier'ını ekler", () => {
    const { container } = render(
      <StatusCard
        status={{ state: "AWAKE", eyes_closed: true, closed_duration: 0.8 }}
        connectionError={null}
        isLoading={false}
      />
    );

    expect(container.querySelector(".driver-state-approaching")).toBeInTheDocument();
  });

  it("DROWSY iken 'yaklaşıyor' modifier'ı eklenmez (zaten gerçek alarm durumu)", () => {
    const { container } = render(
      <StatusCard
        status={{ state: "DROWSY", eyes_closed: true, closed_duration: 3.0 }}
        connectionError={null}
        isLoading={false}
      />
    );

    expect(container.querySelector(".driver-state-approaching")).not.toBeInTheDocument();
    expect(container.querySelector(".driver-state-drowsy")).toBeInTheDocument();
  });

  it("gözler açıkken 'yaklaşıyor' modifier'ı eklenmez", () => {
    const { container } = render(
      <StatusCard
        status={{ state: "AWAKE", eyes_closed: false, closed_duration: 0 }}
        connectionError={null}
        isLoading={false}
      />
    );

    expect(container.querySelector(".driver-state-approaching")).not.toBeInTheDocument();
  });

  it("UNVERIFIABLE durumunda gözleri 'Açık' diye göstermez", () => {
    // Ölçüm alınamadığında "Açık" yazmak, sistemin göremediği bir şeyi
    // gördüğünü iddia etmesi olurdu — bu ekranın en kritik davranışı.
    render(
      <StatusCard
        status={{ state: "UNVERIFIABLE", eyes_closed: false, measurement_valid: false }}
        connectionError={null}
        isLoading={false}
      />,
    );

    expect(screen.getByText("Bilinmiyor")).toBeInTheDocument();
    expect(screen.queryByText("Açık")).not.toBeInTheDocument();
  });

  it("UNVERIFIABLE durumunda kullanıcıya ne yapması gerektiğini söyler", () => {
    render(
      <StatusCard status={{ state: "UNVERIFIABLE" }} connectionError={null} isLoading={false} />,
    );

    expect(screen.getByText(/yüzünüzü kameraya dönün/i)).toBeInTheDocument();
  });

  it("measurement_valid alanı yokken (mediapipe backend) göz durumunu normal gösterir", () => {
    render(
      <StatusCard
        status={{ state: "AWAKE", eyes_closed: false }}
        connectionError={null}
        isLoading={false}
      />,
    );

    expect(screen.getByText("Açık")).toBeInTheDocument();
  });
});
