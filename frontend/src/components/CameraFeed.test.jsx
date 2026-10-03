import { fireEvent, render, screen } from "@testing-library/react";
import { describe, expect, it } from "vitest";
import CameraFeed from "./CameraFeed";

describe("CameraFeed", () => {
  it("başlangıçta kamera görüntüsünü gösterir", () => {
    render(<CameraFeed />);

    expect(screen.getByAltText(/canlı uyku tespit kamerası/i)).toBeInTheDocument();
  });

  it("görüntü yüklenemezse tekrar dene butonunu gösterir", () => {
    render(<CameraFeed />);

    fireEvent.error(screen.getByAltText(/canlı uyku tespit kamerası/i));

    expect(screen.getByRole("button", { name: /tekrar dene/i })).toBeInTheDocument();
    expect(screen.queryByAltText(/canlı uyku tespit kamerası/i)).not.toBeInTheDocument();
  });

  it("tekrar dene tıklanınca kamera görüntüsünü yeniden gösterir", () => {
    render(<CameraFeed />);

    fireEvent.error(screen.getByAltText(/canlı uyku tespit kamerası/i));
    fireEvent.click(screen.getByRole("button", { name: /tekrar dene/i }));

    expect(screen.getByAltText(/canlı uyku tespit kamerası/i)).toBeInTheDocument();
    expect(screen.queryByRole("button", { name: /tekrar dene/i })).not.toBeInTheDocument();
  });

  it("status verilmediyse overlay pill'leri hiç render etmez", () => {
    const { container } = render(<CameraFeed />);

    expect(container.querySelector(".camera-overlay")).not.toBeInTheDocument();
  });

  it("AWAKE durumunda güvenli (yeşil) pill ve FPS'i gösterir", () => {
    render(<CameraFeed status={{ state: "AWAKE", fps: 29.73 }} />);

    const pill = screen.getByText("Uyanık");
    expect(pill).toHaveClass("safe");
    expect(screen.getByText("29.7 FPS")).toBeInTheDocument();
  });

  it("DROWSY durumunda tehlike (kırmızı) pill gösterir", () => {
    render(<CameraFeed status={{ state: "DROWSY", fps: 12 }} />);

    expect(screen.getByText("Uyku hali")).toHaveClass("danger");
  });

  it("stream hata verince overlay pill'leri de kaybolur", () => {
    render(<CameraFeed status={{ state: "AWAKE", fps: 30 }} />);

    fireEvent.error(screen.getByAltText(/canlı uyku tespit kamerası/i));

    expect(screen.queryByText("Uyanık")).not.toBeInTheDocument();
  });
});
