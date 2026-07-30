# Project Rules: HarmoniQ (Music Separator & EQ/Compressor)

ไฟล์นี้กำหนดกฎและสไตล์ไกด์ไลน์สำหรับ AI Agents (รวมถึง opencode) ที่เข้ามาพัฒนาหรือแก้ไขโค้ดในโปรเจกต์นี้

---

## 1. ข้อมูลภาพรวมโปรเจกต์ (Project Overview)
* **ชื่อโปรเจกต์:** HarmoniQ (ระบบแยกแทร็กเสียงดนตรีและปรับแต่งเสียงด้วย AI)
* **สถาปัตยกรรม:**
  * **Frontend:** Next.js (App Router), React 19, TypeScript, Tailwind CSS 4, WaveSurfer.js, NextAuth v4
  * **Database:** SQLite via Prisma ORM
  * **Backend:** FastAPI (Python 3.10), PyTorch, Librosa, Open-Unmix, Pedalboard
  * **Testing:** Jest + ts-jest (Frontend), pytest + unittest (Backend)
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
5. **การตรวจสอบความถูกต้องของโค้ด (Verification):**
   * รัน `pytest` หรือ `python -m unittest discover -s backend/tests` เพื่อทดสอบ Backend ก่อน push หรือ merge ทุกครั้ง

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
   * รายการ Environment Variables ทั้งหมดที่ใช้ในโปรเจกต์:

     | ตัวแปร | คำอธิบาย |
     |--------|----------|
     | `DATABASE_URL` | SQLite database path (`file:./dev.db`) |
     | `NEXT_PUBLIC_API_BASE` | Backend API URL (`http://localhost:8000`) |
     | `NEXTAUTH_SECRET` | Secret key สำหรับ NextAuth session |
     | `NEXTAUTH_URL` | Frontend URL (`http://localhost:3000`) |
     | `SEPARATE_TTL_SECONDS` | TTL cleanup (default: 1200) |
     | `CLEANUP_INTERVAL_SECONDS` | Cleanup interval (default: 300) |
     | `MAX_CONCURRENT_TASKS` | Max concurrent tasks (default: 2) |
5. **การตรวจสอบความถูกต้องของโค้ด (Verification):**
   * ห้ามรัน `npm run build` ในเบื้องหลังขณะที่ Dev Server (`npm run dev`) กำลังทำงานอยู่ เพราะจะไปเขียนทับโฟลเดอร์ `.next` จนเกิดข้อผิดพลาด `500 ENOENT` ในระหว่างพัฒนา ให้ใช้คำสั่ง `npm run type-check` (`npx tsc --noEmit`) หรือ `npm run lint` แทนเสมอ
6. **การทดสอบ (Testing):**
   * รัน `npx jest` สำหรับ frontend tests เพื่อตรวจสอบว่าโค้ดที่แก้ไขไม่ทำลายฟังก์ชันการทำงานเดิม

---

## 3. กฎสำหรับการทำเวอร์ชันคอนโทรล (Git Rules)
* **ห้าม Commit โฟลเดอร์ `.venv/` และโหนดอื่น ๆ ที่ติดตั้งในเครื่อง** (เช่น `node_modules/`, output folders)
* **ห้าม Commit ไฟล์ฐานข้อมูล SQLite** (`*.db`, `*.db-journal`, `prisma/dev.db`)
* ไฟล์ล็อกต่าง ๆ (เช่น `.git/index.lock`) หากเกิดค้าง สามารถใช้คำสั่ง `Remove-Item -Force ".git/index.lock"` เพื่อเคลียร์ไฟล์ได้
* ห้าม Commit ไฟล์เสียงตัวอย่าง หรือไฟล์ขนาดใหญ่เกินกว่า 50MB ขึ้น GitHub โดยเด็ดขาด

---

## 4. การใช้เครื่องมือเสริม (Project Skills Guidelines)
* ทุกครั้งที่ทำงานหรือตอบคำถาม ให้พิจารณาและนำแนวทางของ **Skills** ในโฟลเดอร์ `.agents/skills/` มาปรับใช้โดยอัตโนมัติ ตามความเหมาะสมของหน้างาน:

  **🏗️ การวางแผน & สถาปัตยกรรม**
  * **brainstorming:** ใช้ก่อนเริ่มงานสร้างสรรค์เพื่อสำรวจความต้องการและออกแบบก่อนลงมือทำ
  * **writing-plans:** ใช้เมื่อมี spec หรือ requirements สำหรับงานหลายขั้นตอน ก่อนเริ่มเขียนโค้ด
  * **executing-plans:** ใช้เมื่อมีแผนการดำเนินงานที่ต้องปฏิบัติในเซสชันแยก พร้อม checkpoint
  * **improve-codebase-architecture:** ตรวจสอบโครงสร้างและการจัดวางโมดูล/ฟังก์ชันให้เป็นสัดส่วน
  * **subagent-driven-development:** ใช้ดำเนินการตามแผนโดยส่ง implementer agent ต่อหนึ่งงาน

  **🐛 การพัฒนา & แก้ไข**
  * **test-driven-development:** เขียนเทสก่อน เขียนโค้ดให้ผ่านเทส
  * **systematic-debugging:** ใช้เมื่อพบ bug หรือ test failure ก่อนเสนอแนวทางแก้ไข
  * **ponytail:** เขียนโค้ดให้สั้น กระชับ เรียบง่ายที่สุด ตัดของที่ไม่จำเป็นทิ้ง
  * **dispatching-parallel-agents:** ใช้เมื่อมี 2+ งานอิสระที่ทำพร้อมกันได้

  **👁️ การตรวจสอบ & รีวิว**
  * **requesting-code-review:** ใช้เมื่อทำงานเสร็จหรือก่อน merge เพื่อส่งตรวจสอบ
  * **receiving-code-review:** ใช้เมื่อได้รับคำแนะนำจาก Code Review ก่อนนำไปปรับใช้
  * **verification-before-completion:** ต้องรันคำสั่งตรวจสอบและยืนยันผลลัพธ์ก่อนสรุปงาน
  * **vercel-react-best-practices:** ตรวจสอบประสิทธิภาพการทำ rendering ของ React/Next.js

  **🧪 การทดสอบ**
  * **testsprite-onboard:** ตั้งค่า TestSprite ครั้งแรกใน repository
  * **testsprite-verify:** รันเทสหลังจากทำฟีเจอร์หรือแก้ไขเสร็จ

  **🎨 หน้าจอ & ดีไซน์**
  * **frontend-design:** ทุกครั้งที่มีการแก้ไขหน้าจอ คอนเซ็ปต์ดีไซน์ หรือสไตล์ (UI/UX)
  * **agent-browser:** ใช้เมื่อต้องการเปิดเบราว์เซอร์เพื่อทดสอบหรือจำลองการกระทำผู้ใช้

  **🔄 Git & การปิดงาน**
  * **using-git-worktrees:** ใช้เมื่อเริ่มงานฟีเจอร์ใหม่ที่ต้องการแยก environment
  * **finishing-a-development-branch:** ใช้เมื่อพัฒนาจบและต้องตัดสินใจ merge/PR/cleanup

  **⚡ อื่น ๆ**
  * **using-superpowers:** แนวทางปฏิบัติการพัฒนาซอฟต์แวร์แบบอิงตามแผนงาน
  * **writing-skills:** ใช้เมื่อสร้าง skill ใหม่ แก้ไข หรือตรวจสอบ skill

