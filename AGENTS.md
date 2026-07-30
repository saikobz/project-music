# Project Rules: HarmoniQ (Music Separator & EQ/Compressor)

ไฟล์นี้กำหนดกฎและสไตล์ไกด์ไลน์สำหรับ AI Agents (รวมถึง Antigravity) ที่เข้ามาพัฒนาหรือแก้ไขโค้ดในโปรเจกต์นี้

---

## 1. ข้อมูลภาพรวมโปรเจกต์ (Project Overview)
* **ชื่อโปรเจกต์:** HarmoniQ (ระบบแยกแทร็กเสียงดนตรีและปรับแต่งเสียงด้วย AI)
* **สถาปัตยกรรม:**
  * **Frontend:** Next.js (App Router), React 19, TypeScript, Tailwind CSS 4, WaveSurfer.js
  * **Backend:** FastAPI (Python 3.10), PyTorch, Librosa, Open-Unmix
  * **การสื่อสาร:** Frontend เรียกใช้ Backend API ผ่านพอร์ต `8000` (ควบคุมผ่าน `NEXT_PUBLIC_API_BASE`)

---

## 2. กฎและคำแนะนำทางเทคนิค (Technical Rules)

### 🐍 Backend (FastAPI / Python 3.10)
1. **Python Version Compatibility:**
   * ห้ามใช้ฟีเจอร์ใหม่ที่เริ่มมีใน Python 3.11 หรือสูงกว่า (เช่น `ExceptionGroup`, `typing.Self` หรือไวยากรณ์ Type Parameter แบบใหม่ของ 3.12) เนื่องจากรันบน **Python 3.10**
2. **ภาษาของคำอธิบายโค้ด (Comment Language):**
   * โค้ดเดิมเขียนคำอธิบาย (Comments) และ Docstring ส่วนใหญ่เป็น **ภาษาไทย** ขอให้เขียนคอมเมนต์อธิบายโค้ดภาษาไทยในการแก้ไขโค้ดด้วยเช่นกัน เพื่อความเป็นเอกภาพ
3. **การจัดการไฟล์เสียง:**
   * รองรับไฟล์อินพุตเฉพาะรูปแบบ **WAV (.wav)** ขนาดไม่เกิน **100MB** เท่านั้น
   * ตรวจสอบว่าไฟล์ชั่วคราวทั้งหมดที่สร้างในโฟลเดอร์ `uploads/`, `separated/`, `eq_applied/`, หรือ `compressed/` มีการตั้งค่าลบอัตโนมัติ (Cleanup Task) เสมอเพื่อประหยัดพื้นที่ดิสก์
4. **การจัดการทรัพยากร:**
   * ตรวจสอบว่าได้ทำการปิดออบเจกต์ไฟล์เสียง (เช่น `soundfile`) หรือล้างหน่วยความจำของ PyTorch (ถ้าเป็นไปได้) เพื่อป้องกันปัญหา Memory Leak

### ⚛️ Frontend (Next.js / React 19 / TS)
1. **โครงสร้างโฟลเดอร์:**
   * หน้าเว็บหลักและเลย์เอาต์หลักอยู่ใน `app/` (Next.js App Router)
   * คอมโพเนนต์ย่อยสำหรับเล่นเสียงและการอัปโหลดอยู่ใน `app/components/`
2. **การจัดสไตล์ (Styling):**
   * ใช้ **Tailwind CSS 4** ในการจัดการหน้าจอและสไตล์ ห้ามใช้ inline styles ที่ไม่จำเป็น
   * รักษาหน้าตาเว็บที่ดูดี (Premium Aesthetics) รองรับ Responsive Design ทุกขนาดหน้าจอ
3. **เครื่องเล่นเสียง (Audio Player):**
   * การแสดงผล Waveform และเครื่องเล่นหลายแทร็กใช้ไลบรารี **WaveSurfer.js** หากแก้ไขโค้ดที่เกี่ยวข้อง ให้ตรวจสอบเรื่องการผูก Event Listener และการ Clean up Instance เสมอเพื่อป้องกัน Memory Leak
4. **Environment Variables:**
   * ตัวแปรที่จะดึงไปใช้ฝั่ง Browser เสมอต้องมีพรีฟิกซ์ขึ้นต้นด้วย `NEXT_PUBLIC_` เช่น `NEXT_PUBLIC_API_BASE`
5. **การตรวจสอบความถูกต้องของโค้ด (Verification):**
   * ห้ามรัน `npm run build` ในเบื้องหลังขณะที่ Dev Server (`npm run dev`) กำลังทำงานอยู่ เพราะจะไปเขียนทับโฟลเดอร์ `.next` จนเกิดข้อผิดพลาด `500 ENOENT` ในระหว่างพัฒนา ให้ใช้คำสั่ง `npm run type-check` (`npx tsc --noEmit`) หรือ `npm run lint` แทนเสมอ

---

## 3. กฎสำหรับการทำเวอร์ชันคอนโทรล (Git Rules)
* **ห้าม Commit โฟลเดอร์ `.venv/` และโหนดอื่น ๆ ที่ติดตั้งในเครื่อง** (เช่น `node_modules/`, output folders)
* ไฟล์ล็อกต่าง ๆ (เช่น `.git/index.lock`) หากเกิดค้าง สามารถใช้คำสั่ง `Remove-Item -Force ".git/index.lock"` เพื่อเคลียร์ไฟล์ได้
* ห้าม Commit ไฟล์เสียงตัวอย่าง หรือไฟล์ขนาดใหญ่เกินกว่า 50MB ขึ้น GitHub โดยเด็ดขาด

---

## 4. การใช้เครื่องมือเสริม (Project Skills Guidelines)
* ทุกครั้งที่ทำงานหรือตอบคำถาม ให้พิจารณาและนำแนวทางของ **Skills** ในโฟลเดอร์ `.agents/skills/` มาปรับใช้โดยอัตโนมัติ ตามความเหมาะสมของหน้างาน:
  * **frontend-design:** ทุกครั้งที่มีการแก้ไขหน้าจอ คอนเซ็ปต์ดีไซน์ หรือสไตล์ (UI/UX)
  * **ponytail:** ใช้เขียนโค้ดให้สั้น กระชับ และเรียบง่ายที่สุด โดยตัดโค้ดที่ซ้ำซ้อนหรือไม่จำเป็นออก
  * **vercel-react-best-practices:** นำมาตรวจสอบคุณภาพประสิทธิภาพการทำ rendering ของหน้า React/Next.js
  * **improve-codebase-architecture:** ตรวจสอบโครงสร้างและการจัดวางโมดูล/ฟังก์ชันให้เป็นสัดส่วน
  * **agent-browser:** เรียกใช้งานเมื่อต้องการเปิดเบราว์เซอร์เพื่อทำการเข้าชมเว็บ สตรีมการทดสอบ หรือจำลองการกระทำจริงของผู้ใช้
  * **using-superpowers:** ใช้แนวทางปฏิบัติการพัฒนาซอฟต์แวร์แบบอิงตามแผนงาน (TDD, แผนการแก้ไขอย่างเป็นระบบ และการแยก subagent)

