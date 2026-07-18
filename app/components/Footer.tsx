// Footer ที่ใช้ร่วมกันสำหรับทุกหน้าย่อย
export function Footer() {
  return (
    <footer className="border-t border-[#1E1E1E] py-6 mt-auto">
      <div className="mx-auto max-w-7xl px-4 flex flex-col sm:flex-row items-center justify-between gap-2">
        <p className="text-xs text-[#444444]">© 2025 HarmoniQ — AI Audio Toolkit</p>
        <p className="text-xs text-[#333333]">Built with Next.js · FastAPI · PyTorch · Open-Unmix</p>
      </div>
    </footer>
  );
}
