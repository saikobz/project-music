// lib/download.ts
// ดาวน์โหลดไฟล์ผ่าน fetch -> blob (M9: ลิงก์ `<a href={backend}>` ข้าม origin
// ไม่ทำงานกับ `download` attribute และจะ navigate ไปหน้า 404 เมื่อไฟล์หมด TTL)

export async function downloadViaBlob(url: string, filename: string): Promise<boolean> {
  try {
    const res = await fetch(url);
    if (!res.ok) {
      return false;
    }
    const blob = await res.blob();
    const objectUrl = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = objectUrl;
    a.download = filename;
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
    URL.revokeObjectURL(objectUrl);
    return true;
  } catch {
    return false;
  }
}
