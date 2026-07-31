/**
 * regression tests สำหรับ TU-7 (Frontend Quota Flow):
 * - M8: เลือกไฟล์ใหม่ -> ผลลัพธ์เก่า (blob URL, fileId, analysis) ต้องถูกล้าง
 * - F11: quota 403 -> หยุดก่อนส่งไฟล์ + ไม่มีการ refund (ยังไม่ได้หัก)
 * - F11: ประมวลผลล้มเหลว -> เรียก /api/quota/refund เพื่อคืนโควตา
 * - F12: SingleExportModal เปลี่ยน format -> ไม่เรียก /api/quota/consume ซ้ำ
 *
 * @jest-environment jsdom
 */
import { act } from "react";
import { createRoot, Root } from "react-dom/client";

(globalThis as { IS_REACT_ACT_ENVIRONMENT?: boolean }).IS_REACT_ACT_ENVIRONMENT = true;

jest.mock("axios", () => ({
  __esModule: true,
  default: {
    isCancel: jest.fn(() => false),
    post: jest.fn(),
  },
}));

jest.mock("sonner", () => ({
  toast: { success: jest.fn(), error: jest.fn() },
}));

jest.mock("next-auth/react", () => ({
  useSession: jest.fn(() => ({ data: null, status: "unauthenticated" })),
}));

jest.mock("../app/components/WaveformPlayer", () => () => <div data-testid="waveform-player" />);
jest.mock("../app/components/MultiStemLivePlayer", () => () => <div data-testid="multi-stem-player" />);
jest.mock("../app/components/AudioAnalysis", () => () => <div data-testid="audio-analysis" />);
// ตั้งค่า settings เป็น named exports (import แบบ { AutoEqSettings })
jest.mock("../app/components/settings/AutoEqSettings", () => ({
  AutoEqSettings: () => <div />,
}));
jest.mock("../app/components/settings/CompressorSettings", () => ({
  CompressorSettings: () => <div />,
}));
jest.mock("../app/components/settings/PitchShiftSettings", () => ({
  PitchShiftSettings: () => <div />,
}));
jest.mock("../app/components/ExportMasterModal", () => () => <div />);
jest.mock("../app/components/SingleExportModal", () => {
  const SingleExportModal = ({ isOpen, onExport }: any) =>
    isOpen ? <button data-testid="export-modal-btn" onClick={() => onExport("mp3")}>Export mp3</button> : null;
  return SingleExportModal;
});

import axios from "axios";
import { useSession } from "next-auth/react";
import UploadBox from "../app/components/UploadBox";

const axiosMock = axios as unknown as { post: jest.Mock; isCancel: jest.Mock };
const sessionMock = useSession as jest.Mock;

let container: HTMLDivElement;
let root: Root;

const WAV_FILE = new File([new Uint8Array([0, 0, 0, 0])], "song.wav", { type: "audio/wav" });

function renderUploadBox() {
  container = document.createElement("div");
  document.body.appendChild(container);
  root = createRoot(container);
  act(() => {
    root.render(<UploadBox />);
  });
}

function selectFile(file: File = WAV_FILE) {
  let input = container.querySelector('input[type="file"]') as HTMLInputElement | null;
  // ถ้าเลือกไฟล์ไปแล้ว input จะหายจาก DOM (สลับเป็นหน้าประมวลผล) -> กด Change File ก่อน
  if (!input) {
    const buttons = Array.from(container.querySelectorAll("button"));
    const changeBtn = buttons.find((b) => b.textContent?.includes("Change File")) as HTMLButtonElement;
    act(() => {
      changeBtn.click();
    });
    input = container.querySelector('input[type="file"]') as HTMLInputElement;
  }
  act(() => {
    Object.defineProperty(input, "files", { value: [file], configurable: true });
    input.dispatchEvent(new Event("change", { bubbles: true }));
  });
}

// เปลี่ยน action เป็น Pitch Shift (ต้องมี blob URL ผลลัพธ์) — default คือ "separate"
function selectPitchAction() {
  const buttons = Array.from(container.querySelectorAll("button"));
  const tab = buttons.find((b) => b.textContent?.includes("Pitch Shift")) as HTMLButtonElement;
  act(() => {
    tab.click();
  });
}

async function processPitchFile() {
  selectFile(WAV_FILE);
  selectPitchAction();
  clickProcess();
  await act(async () => {
    await Promise.resolve();
    await Promise.resolve();
    await Promise.resolve();
  });
}

function clickProcess() {
  const processBtn = container.querySelector('[data-testid="process-button"]') as HTMLButtonElement;
  act(() => {
    processBtn.click();
  });
}

function getDownloadButton() {
  const buttons = Array.from(container.querySelectorAll("button"));
  return buttons.find((b) => b.textContent?.includes("Export & Download")) as
    | HTMLButtonElement
    | undefined;
}

describe("UploadBox (TU-7)", () => {
  let revokeObjectURLMock: jest.Mock;
  let fetchMock: jest.Mock;

  beforeEach(() => {
    jest.clearAllMocks();
    // jsdom ไม่มี URL.createObjectURL / global fetch -> ต้อง mock เอง
    (URL as unknown as { createObjectURL: jest.Mock }).createObjectURL = jest
      .fn()
      .mockReturnValue("blob:mock-url");
    revokeObjectURLMock = jest.fn();
    (URL as unknown as { revokeObjectURL: jest.Mock }).revokeObjectURL = revokeObjectURLMock;
    fetchMock = jest.fn().mockResolvedValue({
      ok: true,
      status: 200,
      json: async () => ({ success: true }),
      blob: async () => new Blob(["RIFF"]),
    });
    (globalThis as unknown as { fetch: jest.Mock }).fetch = fetchMock;
    axiosMock.post.mockResolvedValue({
      data: new Blob(["RIFF"]),
      status: 200,
    });
    renderUploadBox();
  });

  afterEach(() => {
    act(() => root.unmount());
    container.remove();
  });

  it("selecting a new file clears previous results and revokes blob URL (M8)", async () => {
    // ประมวลผลไฟล์แรกสำเร็จ (pitch shift -> blob URL ถูกสร้าง)
    await processPitchFile();
    expect(
      (URL as unknown as { createObjectURL: jest.Mock }).createObjectURL
    ).toHaveBeenCalled();
    expect(getDownloadButton()).toBeDefined();

    // เลือกไฟล์ใหม่ -> blob URL เก่าต้องถูก revoke
    const fileB = new File([new Uint8Array([1, 1, 1, 1])], "song2.wav", { type: "audio/wav" });
    selectFile(fileB);

    expect(revokeObjectURLMock).toHaveBeenCalledWith("blob:mock-url");
    // แผงผลลัพธ์เก่า (ปุ่ม Export & Download) ต้องหายไป
    expect(getDownloadButton()).toBeUndefined();
  });

  it("does not send file to backend or refund when quota returns 403 (F11)", async () => {
    sessionMock.mockReturnValue({ data: { user: { id: "user-1" } }, status: "authenticated" });
    // quota 403
    fetchMock.mockResolvedValueOnce({
      ok: false,
      status: 403,
      json: async () => ({ error: "quota full" }),
    });

    selectFile(WAV_FILE);
    clickProcess();
    await act(async () => {
      await Promise.resolve();
      await Promise.resolve();
    });

    expect(axiosMock.post).not.toHaveBeenCalled();
    expect(fetchMock).not.toHaveBeenCalledWith("/api/quota/refund", expect.anything());
  });

  it("refunds quota when processing fails after quota was charged (F11)", async () => {
    sessionMock.mockReturnValue({ data: { user: { id: "user-1" } }, status: "authenticated" });
    // quota consume สำเร็จ (200)
    fetchMock.mockResolvedValue({ ok: true, status: 200, json: async () => ({ success: true }) });
    // backend ล้มเหลว
    axiosMock.post.mockRejectedValue({ message: "boom", code: "ERR_BAD_RESPONSE" });

    selectFile(WAV_FILE);
    clickProcess();
    await act(async () => {
      await Promise.resolve();
      await Promise.resolve();
    });

    expect(fetchMock).toHaveBeenCalledWith("/api/quota/refund", { method: "POST" });
    expect(axiosMock.post).toHaveBeenCalled();
  });

  it("blocks submit when trim values are invalid (M14)", async () => {
    selectFile(WAV_FILE);

    // เปิด Trim (click label เหมือน user จริง — span ข้างในมี onClick toggle)
    const labels = Array.from(container.querySelectorAll("label"));
    const trimLabel = labels.find((l) => l.textContent?.includes("Trim"));
    act(() => {
      trimLabel?.click();
    });
    const numberInputs = Array.from(container.querySelectorAll('input[type="number"]')) as HTMLInputElement[];
    // ลำดับ: [0]=trimStart, [1]=trimEnd (action default=separate ไม่มี pitch input)
    const setNativeValue = (el: HTMLInputElement, value: string) => {
      const setter = Object.getOwnPropertyDescriptor(window.HTMLInputElement.prototype, "value")?.set;
      setter?.call(el, value);
      el.dispatchEvent(new Event("input", { bubbles: true }));
    };
    act(() => {
      setNativeValue(numberInputs[0], "5");
    });
    act(() => {
      setNativeValue(numberInputs[1], "2");
    });

    clickProcess();
    await act(async () => {
      await Promise.resolve();
      await Promise.resolve();
    });

    expect(axiosMock.post).not.toHaveBeenCalled();
    const errorText = container.textContent || "";
    expect(errorText).toContain("trim_end");
  });

  it("does not consume quota twice and converts without re-processing (F12)", async () => {
    sessionMock.mockReturnValue({ data: { user: { id: "user-1" } }, status: "authenticated" });
    fetchMock.mockResolvedValue({
      ok: true,
      status: 200,
      json: async () => ({ success: true }),
      blob: async () => new Blob(["RIFF"]),
    });

    // ประมวลผลครั้งแรก (consume 1 ครั้ง)
    await processPitchFile();
    const consumeCalls = () =>
      fetchMock.mock.calls.filter((c) => c[0] === "/api/quota/consume");
    expect(consumeCalls()).toHaveLength(1);

    // เปิด SingleExportModal แล้วเลือก format ต่าง (mp3)
    const exportBtn = getDownloadButton() as HTMLButtonElement;
    act(() => {
      exportBtn.click();
    });
    const modalBtn = container.querySelector('[data-testid="export-modal-btn"]') as HTMLButtonElement;
    await act(async () => {
      modalBtn.click();
      await Promise.resolve();
      await Promise.resolve();
      await Promise.resolve();
    });

    // ไม่เรียก quota ซ้ำ และไม่ re-process ไฟล์เดิม (ไป /convert-format แทน)
    expect(consumeCalls()).toHaveLength(1);
    const convertCalls = axiosMock.post.mock.calls.filter((c) =>
      String(c[0]).includes("/convert-format")
    );
    expect(convertCalls).toHaveLength(1);
    const reprocessCalls = axiosMock.post.mock.calls.filter((c) =>
      String(c[0]).includes("/pitch-shift")
    );
    expect(reprocessCalls).toHaveLength(1);
  });
});
