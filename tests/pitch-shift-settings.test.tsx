/**
 * regression tests สำหรับ A1 (M12): พิมพ์ pitch ติดลบได้
 * - พิมพ์ "-" ระหว่างทางต้องไม่ commit ค่า 0 ให้ parent
 * - พิมพ์ "-2" ครบ -> setPitchSteps(-2)
 *
 * หมายเหตุ: jsdom sanitize "-" บน <input type=number> เป็น "" (บราวเซอร์จริงเก็บ "-" ไว้)
 * ดังนั้น test นี้จึงตรวจสาระสำคัญคือ "ไม่ commit 0 ระหว่างพิมพ์" แทน
 *
 * @jest-environment jsdom
 */
import { act } from "react";
import { createRoot, Root } from "react-dom/client";

import { PitchShiftSettings } from "../app/components/settings/PitchShiftSettings";

(globalThis as { IS_REACT_ACT_ENVIRONMENT?: boolean }).IS_REACT_ACT_ENVIRONMENT = true;

let container: HTMLDivElement;
let root: Root;
let setPitchSteps: jest.Mock;

function renderComponent(pitchSteps = 0) {
  container = document.createElement("div");
  document.body.appendChild(container);
  root = createRoot(container);
  act(() => {
    root.render(
      <PitchShiftSettings
        pitchSteps={pitchSteps}
        setPitchSteps={setPitchSteps}
        loading={false}
      />
    );
  });
}

function typeIn(value: string) {
  const input = container.querySelector('[data-testid="pitch-input"]') as HTMLInputElement;
  act(() => {
    // ใช้ native setter ของ HTMLInputElement เพื่อให้ React value tracker เห็นการเปลี่ยนค่า
    const setter = Object.getOwnPropertyDescriptor(window.HTMLInputElement.prototype, "value")?.set;
    setter?.call(input, value);
    input.dispatchEvent(new Event("input", { bubbles: true }));
  });
  return input;
}

describe("PitchShiftSettings (A1 M12)", () => {
  beforeEach(() => {
    setPitchSteps = jest.fn();
  });

  afterEach(() => {
    act(() => root.unmount());
    container.remove();
  });

  it("does not commit 0 while typing '-' (partial input)", () => {
    renderComponent();
    typeIn("-");
    // jsdom sanitize "-" -> "" แต่ไม่ว่ากรณีใด ต้องไม่เรียก setPitchSteps(0)
    expect(setPitchSteps).not.toHaveBeenCalled();
  });

  it("commits -2 once typing completes", () => {
    renderComponent();
    typeIn("-2");
    expect(setPitchSteps).toHaveBeenCalledWith(-2);
  });

  it("commits positive numbers normally", () => {
    renderComponent();
    typeIn("3");
    expect(setPitchSteps).toHaveBeenCalledWith(3);
  });

  it("syncs display value when parent changes pitchSteps externally", () => {
    setPitchSteps.mockImplementationOnce(() => {});
    renderComponent(0);
    // จำลอง parent เปลี่ยนค่าเป็น 5 (เช่น clamp จากภายนอก)
    act(() => {
      root.render(
        <PitchShiftSettings pitchSteps={5} setPitchSteps={setPitchSteps} loading={false} />
      );
    });
    expect((container.querySelector('[data-testid="pitch-input"]') as HTMLInputElement).value).toBe("5");
  });
});
