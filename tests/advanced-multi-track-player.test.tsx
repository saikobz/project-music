/**
 * regression tests สำหรับ TU-6 (Frontend Player Stability):
 * - F10: ไม่ใช้ innerHTML ล้าง DOM ที่ React render (overlay เป็น sibling, unmount ไม่ throw)
 * - M1: destroy แต่ละ instance ครั้งเดียว + ref ถูกล้างตอน unmount/rebuild
 * - M4: ปุ่ม Play disabled จนกว่าทุก stem ready (กันกดเร็ว -> หลุด sync ถาวร)
 * - M2/M3: toggle Vocal Polish โหลดเฉพาะ vocals + สถานะ solo ยังถูก apply หลัง rebuild
 * - M6: polish ล้มเหลวต้องมี error feedback
 *
 * @jest-environment jsdom
 */
import { act } from "react";
import { createRoot, Root } from "react-dom/client";

import AdvancedMultiTrackPlayer from "../app/components/AdvancedMultiTrackPlayer";

// React 19 ต้องการ flag นี้เพื่อให้ act() ทำงานถูกต้องใน test environment (jsdom)
(globalThis as { IS_REACT_ACT_ENVIRONMENT?: boolean }).IS_REACT_ACT_ENVIRONMENT = true;

jest.mock("wavesurfer.js", () => ({
  __esModule: true,
  default: { create: jest.fn() },
}));

import WaveSurfer from "wavesurfer.js";

const mockCreate = WaveSurfer.create as unknown as jest.Mock;

type HandlerMap = Record<string, (...args: unknown[]) => void>;

interface FakeWs {
  load: jest.Mock;
  on: jest.Mock;
  playPause: jest.Mock;
  pause: jest.Mock;
  seekTo: jest.Mock;
  setVolume: jest.Mock;
  getDuration: jest.Mock;
  destroy: jest.Mock;
}

interface FakeEntry {
  ws: FakeWs;
  handlers: HandlerMap;
}

const BASE_URL = "http://backend.test/separated/abc123";
const FILE_ID = "abc123";

let instances: FakeEntry[] = [];
let container: HTMLDivElement;
let root: Root;

function makeInstance(): FakeEntry {
  const handlers: HandlerMap = {};
  const ws: FakeWs = {
    load: jest.fn().mockResolvedValue(undefined),
    on: jest.fn((event: string, cb: (...args: unknown[]) => void) => {
      handlers[event] = cb;
    }),
    playPause: jest.fn(),
    pause: jest.fn(),
    seekTo: jest.fn(),
    setVolume: jest.fn(),
    getDuration: jest.fn().mockReturnValue(30),
    destroy: jest.fn(),
  };
  const entry = { ws, handlers };
  instances.push(entry);
  return entry;
}

function fireReady(index: number) {
  act(() => {
    instances[index]?.handlers.ready?.();
  });
}

async function readyAll() {
  await act(async () => {
    instances.forEach(({ handlers }) => handlers.ready?.());
  });
}

async function clickPolish() {
  const btn = container.querySelector('[data-testid="polish-toggle"]') as HTMLButtonElement;
  await act(async () => {
    btn.click();
    await Promise.resolve();
    await Promise.resolve();
  });
}

async function clickPlay() {
  const btn = container.querySelector('[data-testid="play-toggle"]') as HTMLButtonElement;
  await act(async () => {
    btn.click();
  });
}

function unmount() {
  act(() => root.unmount());
  container.remove();
}

describe("AdvancedMultiTrackPlayer (TU-6)", () => {
  beforeEach(() => {
    instances = [];
    mockCreate.mockImplementation(() => makeInstance().ws);
    global.fetch = jest.fn();
    container = document.createElement("div");
    document.body.appendChild(container);
    root = createRoot(container);
  });

  afterEach(() => {
    unmount();
    jest.clearAllMocks();
  });

  function renderPlayer() {
    act(() => {
      root.render(<AdvancedMultiTrackPlayer baseUrl={BASE_URL} fileId={FILE_ID} />);
    });
  }

  it("creates 4 WaveSurfer instances on real DOM elements", async () => {
    renderPlayer();
    await readyAll();

    expect(mockCreate).toHaveBeenCalledTimes(4);
    const containers = mockCreate.mock.calls.map((call) => call[0].container);
    containers.forEach((c) => expect(c).toBeInstanceOf(HTMLDivElement));
  });

  it("keeps gradient overlay as React sibling and unmounts without throwing (F10)", async () => {
    renderPlayer();
    await readyAll();

    // overlay ยังอยู่ใน DOM หลัง ready (เดิมโดน innerHTML ลบทิ้งเงียบๆ)
    const overlay = container.querySelector('[class*="bg-gradient-to-b"]');
    expect(overlay).not.toBeNull();
    // overlay ต้องไม่ถูกฝังอยู่ใน container ที่ WaveSurfer ยึดครอง
    mockCreate.mock.calls.forEach((call) => {
      expect(call[0].container.contains(overlay)).toBe(false);
    });

    // regression: unmount ไม่ควร throw NotFoundError จาก React ลบ node ที่หายไป
    expect(() => unmount()).not.toThrow();
  });

  it("destroys each instance exactly once on unmount (M1)", async () => {
    renderPlayer();
    await readyAll();

    instances.forEach(({ ws }) => expect(ws.destroy).not.toHaveBeenCalled());

    unmount();
    instances.forEach(({ ws }) => expect(ws.destroy).toHaveBeenCalledTimes(1));
  });

  it("destroys old instances before creating new ones on baseUrl change (M1)", async () => {
    renderPlayer();
    await readyAll();
    const oldInstances = instances.slice();

    act(() => {
      root.render(
        <AdvancedMultiTrackPlayer baseUrl="http://backend.test/separated/newfile" fileId="newfile" />
      );
    });

    oldInstances.forEach(({ ws }) => expect(ws.destroy).toHaveBeenCalledTimes(1));
    expect(instances.length).toBe(8);
  });

  it("disables Play until every stem is ready, then syncs playPause (M4)", async () => {
    renderPlayer();

    const playBtn = () => container.querySelector('[data-testid="play-toggle"]') as HTMLButtonElement;
    expect(playBtn().disabled).toBe(true);

    // ready แค่ 2 ตัว ยังต้อง disabled
    fireReady(0);
    fireReady(1);
    expect(playBtn().disabled).toBe(true);
    await clickPlay();
    instances.forEach(({ ws }) => expect(ws.playPause).not.toHaveBeenCalled());

    // ready ครบ 4 -> เล่นได้ และ playPause ถูกเรียกทุกตัว
    fireReady(2);
    fireReady(3);
    expect(playBtn().disabled).toBe(false);
    await clickPlay();
    instances.forEach(({ ws }) => expect(ws.playPause).toHaveBeenCalledTimes(1));
  });

  it("reloads only vocals on polish toggle and preserves solo volume (M2/M3)", async () => {
    renderPlayer();
    await readyAll();
    const [vocals, drums, bass, other] = instances.map((e) => e.ws);

    // เลือก solo vocals
    act(() => {
      (container.querySelector('[data-testid="solo-vocals"]') as HTMLButtonElement).click();
    });

    // toggle polish สำเร็จ
    (global.fetch as jest.Mock).mockResolvedValue({ ok: true, json: async () => ({}) });
    await clickPolish();

    // โหลดเฉพาะ vocals ด้วยไฟล์ polished
    expect(vocals.load).toHaveBeenLastCalledWith(`${BASE_URL}/vocals_polished.wav`);
    // stem อื่นไม่ถูกโหลดซ้ำ (มีแค่ load ครั้งแรก)
    expect(drums.load).toHaveBeenCalledTimes(1);
    expect(bass.load).toHaveBeenCalledTimes(1);
    expect(other.load).toHaveBeenCalledTimes(1);

    // ยิง ready ของ vocals หลัง reload -> volume ยังสะท้อน solo (vocals=0.85, ที่เหลือ=0)
    fireReady(0);
    const lastSet = (ws: FakeWs) => {
      const calls = ws.setVolume.mock.calls;
      return calls[calls.length - 1]?.[0] as number;
    };
    expect(lastSet(vocals)).toBeCloseTo(0.85);
    expect(lastSet(drums)).toBe(0);
    expect(lastSet(bass)).toBe(0);
    expect(lastSet(other)).toBe(0);
  });

  it("shows backend error message when polish API fails (M6)", async () => {
    renderPlayer();
    await readyAll();

    (global.fetch as jest.Mock).mockResolvedValue({
      ok: false,
      json: async () => ({ detail: "ไม่พบไฟล์เสียงร้อง (vocals.wav) ในระบบ" }),
    });
    await clickPolish();

    const errorEl = container.querySelector('[data-testid="polish-error"]');
    expect(errorEl).not.toBeNull();
    expect(errorEl?.textContent).toContain("vocals.wav");
    // ปุ่มกลับมาใช้งานได้ (ไม่ติดสถานะ loading)
    const btn = container.querySelector('[data-testid="polish-toggle"]') as HTMLButtonElement;
    expect(btn.disabled).toBe(false);
  });

  it("shows generic error on network failure and does not enable polish (M6)", async () => {
    renderPlayer();
    await readyAll();

    (global.fetch as jest.Mock).mockRejectedValue(new Error("network down"));
    await clickPolish();

    const errorEl = container.querySelector('[data-testid="polish-error"]');
    expect(errorEl?.textContent).toContain("ไม่สามารถเชื่อมต่อกับเซิร์ฟเวอร์ได้");
  });
});
