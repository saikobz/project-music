/**
 * regression tests สำหรับ B1 (M9): downloadViaBlob
 * - fetch สำเร็จ -> สร้าง anchor + download + revoke object URL -> true
 * - response !ok (ไฟล์หมด TTL -> 404) -> false
 * - network error -> false
 *
 * @jest-environment jsdom
 */
import { downloadViaBlob } from "../lib/download";

(globalThis as { IS_REACT_ACT_ENVIRONMENT?: boolean }).IS_REACT_ACT_ENVIRONMENT = true;

describe("downloadViaBlob (B1 M9)", () => {
  let fetchMock: jest.Mock;
  let clickSpy: jest.SpyInstance;
  let createObjectURLMock: jest.Mock;
  let revokeObjectURLMock: jest.Mock;

  beforeEach(() => {
    fetchMock = jest.fn();
    (globalThis as unknown as { fetch: jest.Mock }).fetch = fetchMock;
    createObjectURLMock = jest.fn().mockReturnValue("blob:mock");
    (URL as unknown as { createObjectURL: jest.Mock }).createObjectURL = createObjectURLMock;
    revokeObjectURLMock = jest.fn();
    (URL as unknown as { revokeObjectURL: jest.Mock }).revokeObjectURL = revokeObjectURLMock;
    // ปิดเสียง "Not implemented: navigation" จาก jsdom เมื่อ anchor.click()
    clickSpy = jest.spyOn(HTMLAnchorElement.prototype, "click").mockImplementation(() => {});
  });

  afterEach(() => {
    clickSpy.mockRestore();
  });

  it("downloads via blob and returns true on success", async () => {
    fetchMock.mockResolvedValue({
      ok: true,
      blob: async () => new Blob(["zip-data"]),
    });

    const result = await downloadViaBlob("http://backend/download/abc", "separated.zip");

    expect(result).toBe(true);
    expect(createObjectURLMock).toHaveBeenCalled();
    expect(revokeObjectURLMock).toHaveBeenCalledWith("blob:mock");
  });

  it("returns false when response is not ok (file expired -> 404)", async () => {
    fetchMock.mockResolvedValue({ ok: false, status: 404, blob: async () => new Blob() });

    const result = await downloadViaBlob("http://backend/download/expired", "separated.zip");

    expect(result).toBe(false);
    expect(createObjectURLMock).not.toHaveBeenCalled();
  });

  it("returns false on network error", async () => {
    fetchMock.mockRejectedValue(new Error("network"));

    const result = await downloadViaBlob("http://backend/download/abc", "separated.zip");

    expect(result).toBe(false);
  });
});
