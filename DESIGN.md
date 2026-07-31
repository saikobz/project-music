# Design System: HarmoniQ — "Warm Studio"

ไฟล์นี้เป็น Design Specification สำหรับ AI Agents (รวมถึง opencode) ที่ต้องสร้างหรือแก้ไข UI ในโปรเจกต์ HarmoniQ — อ่านไฟล์นี้ก่อนเขียนโค้ด UI เสมอ เพื่อให้การออกแบบสม่ำเสมอตลอดทั้งโปรเจกต์

---

## 1. Brand Identity

| แกน | รายละเอียด |
|---|---|
| **Product** | HarmoniQ — AI-powered audio separation & processing toolkit |
| **Audience** | Music producers, audio engineers, content creators |
| **Mood** | Dark, warm, inviting — เหมือนสตูดิโอบันทึกเสียงระดับพรีเมียมยามค่ำคืน |
| **Personality** | มืออาชีพแต่เข้าถึงง่าย, อบอุ่นไม่เย็นชา, function-first |
| **Inspiration** | Spotify (warmth), Ableton (clarity), high-end audio hardware |
| **Signature** | **Minimal — Function First** ความอบอุ่นมาจาก *สีและฟอนต์* ไม่ใช่ animation หรือ gimmick |

### หลักการออกแบบ (Design Principles)
1. **Function first** — ตัดทุกอย่างที่ไม่จำเป็นออก เครื่องมือต้องอ่านเข้าใจง่าย ใช้เร็ว
2. **ความอบอุ่นผ่านสี** — neutral ทั้งหมดมี warm undertone (แดง/น้ำตาลเจือเล็กน้อย) ไม่ใช้สีเทาดำสนิท
3. **Bold ไว้ที่เดียว** — ใช้ Coral อย่างตั้งใจเฉพาะจุดสำคัญ (CTA, highlight) ส่วนที่เหลือปล่อยให้เรียบ
4. **Restraint** — ไม่ใช้ animation วนเวียน แสง glow แบบ ambient เบา ๆ เท่าที่จำเป็น
5. **Motion สมเหตุสมผล** — animation ทุกตัวต้องสื่อความหมาย เช่น skeleton กำลังโหลด, waveform กำลังเล่น

---

## 2. Color Palette

### 2.1 Neutral Scale (Dark Mode — Warm Undertone)

| Token | Hex | Usage |
|---|---|---|
| `--color-bg-deepest` | `#050505` | Waveform canvas, โค้ด block |
| `--color-bg-root` | `#0D0B0A` | พื้นหลังหน้าเว็บ, modals, navbar |
| `--color-bg-panel` | `#161412` | Cards, drop zones, settings panels |
| `--color-bg-raised` | `#1E1B18` | Hover states, inner cards |
| `--color-border-subtle` | `#1E1E1E` | Navbar border, section dividers |
| `--color-border-standard` | `#2C2824` | Card borders, inputs |
| `--color-border-hover` | `#36322E` | Hover borders, waveform borders |
| `--color-text-primary` | `#F5F0EB` | Headings, primary text |
| `--color-text-secondary` | `#A09890` | Body text, descriptions |
| `--color-text-muted` | `#8E8E8E` | Labels, secondary info |
| `--color-text-dim` | `#5C5854` | Placeholders, disabled text |

### 2.2 Brand Accent — Coral

| Token | Hex | Usage |
|---|---|---|
| `--color-brand` | `#F97316` | Primary CTA buttons, links, active states, focus rings |
| `--color-brand-hover` | `#FB923C` | Hover ของ brand elements |
| `--color-brand-dark` | `#EA580C` | ปลาย gradient ของปุ่มหลัก |
| `--color-brand-soft` | `rgba(249,115,22,0.10)` | Badge backgrounds, soft highlights |
| `--color-brand-border` | `rgba(249,115,22,0.20)` | Badge borders, soft outlines |

### 2.3 Action Colors (ใช้เฉพาะใน Player/Mixer Context เท่านั้น)

สีเหล่านี้บ่งบอกประเภทการประมวลผลเสียง — ห้ามใช้เป็น brand color ทั่วเว็บ

| Action | Hex | Usage |
|---|---|---|
| Separate | `#A78BFA` (ม่วง) | ปุ่ม Separate, vocals stem |
| Auto EQ | `#22D3EE` (ฟ้า) | ปุ่ม Auto EQ |
| Compressor | `#E5A93D` (ทอง) | ปุ่ม Compressor, drums stem |
| Pitch Shift | `#34D399` (เขียว) | ปุ่ม Pitch Shift, bass stem |

**Stem Waveform Colors** (ใน MultiTrack Player):

| Stem | Wave | Progress | Accent |
|---|---|---|---|
| Vocals | `#FBCFE8` | `#F472B6` | `#F9A8D4` |
| Drums | `#FDE68A` | `#F59E0B` | `#FBBF24` |
| Bass | `#A7F3D0` | `#10B981` | `#34D399` |
| Other | `#BFDBFE` | `#38BDF8` | `#7DD3FC` |

### 2.4 Semantic Colors

| Token | Hex | Usage |
|---|---|---|
| Success | `#34D399` | Success toasts, completion states, ตัวเลข quota |
| Error | `#EF4444` | Error messages, destructive actions, confirm delete |
| Warning | `#F59E0B` | Warning toasts, warnings |

---

## 3. Typography

| Role | Font | Usage |
|---|---|---|
| **Display** | **Space Grotesk** | Headings `h1-h3`, big numbers, brand moments |
| **Body** | **Inter** | Paragraphs, buttons, forms, UI text |
| **Utility/Mono** | **Geist Mono** (fallback: `ui-monospace`) | Timecode, dB values, Hz, EQ data |

### Type Scale

| Token | Size | Weight | Usage |
|---|---|---|---|
| `display-xl` | `text-5xl md:text-7xl` | 700 | Homepage hero |
| `display-lg` | `text-4xl md:text-5xl` | 700 | Page headers |
| `heading` | `text-2xl md:text-3xl` | 700 | Section headers |
| `subheading` | `text-xl` | 600 | Card titles |
| `body` | `text-sm md:text-base` | 400 | Paragraphs, form labels |
| `caption` | `text-xs` | 500 | Section eyebrows, labels, badges |
| `data` | `text-xs font-mono` | 500 | Timecode, dB, status text |

### Typography Conventions
- Headings: `tracking-tight` เสมอ
- Eyebrows (label เหนือหัวข้อ): `text-xs font-semibold uppercase tracking-[0.2em]`
- ค่าทางเทคนิค (เวลา, dB, Hz, sample rate): mono font เสมอ
- Line height: display `1.1`, body `1.6`
- ห้ามใช้ `font-light` กับ Space Grotesk — ใช้ `regular/bold` ตามดีไซน์ของตัวฟอนต์
- Sentence case สำหรับ label (ไม่ใช่ ALL CAPS ยกเว้น eyebrow)

---

## 4. Component Patterns

### Card
```tsx
className="rounded-xl border border-[#2C2824] bg-[#161412] p-5 shadow-lg"
```
- Hover: `hover:border-[#36322E] hover:shadow-xl transition-colors duration-200`
- Variant glass: ใช้ utility `glass` ที่มีอยู่ใน `globals.css`

### Primary Button (Coral Brand)
```tsx
className="bg-gradient-to-br from-[#F97316] to-[#EA580C] text-white font-semibold
           rounded-xl px-6 py-3 shadow-[0_4px_20px_rgba(249,115,22,0.25)]
           hover:shadow-[0_6px_25px_rgba(249,115,22,0.35)] transition-all duration-200"
```

### Secondary Button
```tsx
className="rounded-xl border border-[#2C2824] bg-[#161412] px-6 py-3
           text-[#A09890] hover:text-[#F5F0EB] hover:border-[#36322E] transition-colors"
```

### Action Button (เฉพาะใน Player/Mixer — ตัวอย่าง Separate)
```tsx
className="rounded-lg border border-[#A78BFA]/30 bg-[#A78BFA]/10 text-[#A78BFA]
           text-xs font-semibold uppercase tracking-wider
           hover:bg-[#A78BFA]/20 transition"
```

### Input
```tsx
className="w-full rounded-lg bg-[#0D0B0A] border border-[#2C2824] p-2.5
           text-[#F5F0EB] placeholder:text-[#5C5854]
           focus:border-[#F97316] focus:outline-none transition"
```

### Modal
```tsx
// Overlay
className="fixed inset-0 z-50 flex items-center justify-center bg-[#0D0B0A]/80 backdrop-blur-md"
// Inner card
className="rounded-2xl border border-[#2C2824] bg-[#0D0B0A] p-6 shadow-2xl"
```

### Badge / Pill
```tsx
// Brand badge
className="inline-flex items-center gap-2 px-3 py-1.5 rounded-full
           bg-[#F97316]/10 border border-[#F97316]/20 text-[#F97316] text-xs font-medium"
// Action badge (เช่น "Separated")
className="inline-flex items-center gap-1.5 px-2.5 py-1 rounded-md
           bg-[#A78BFA]/10 border border-[#A78BFA]/20 text-[#A78BFA]
           text-[10px] font-semibold uppercase tracking-wider"
```

### Notification / Toast (sonner richColors)
| Type | Background | Border | Text |
|---|---|---|---|
| Success | `bg-[#34D399]/10` | `border-[#34D399]/20` | `text-[#34D399]` |
| Error | `bg-red-950/30` | `border-red-900/50` | `text-red-400` |
| Warning | `bg-[#F59E0B]/10` | `border-[#F59E0B]/20` | `text-[#F59E0B]` |

---

## 5. Animation & Glass Utilities

### Animation Tokens (มีอยู่แล้วใน `globals.css`)

| Token | Duration | Easing | Usage |
|---|---|---|---|
| `fade-in-up` | 0.5s | `cubic-bezier(0.16, 1, 0.3, 1)` | Elements appearing on mount/scroll |
| `shimmer` | 2s | `linear` | Loading skeletons |
| `glow-pulse` | 3s | `ease-in-out` | Ambient background glow (restraint — 1 จุดเท่านั้น) |
| `audiowave` | 1.2s | `ease-in-out` | Animated audio bars ใน player |

### Hover Transitions (มาตรฐาน)
```
transition-colors duration-200   → hover สี/พื้นหลัง
transition-all duration-200      → hover ที่มี shadow + transform
transition-opacity duration-300  → fade in/out
```

### Glass Utilities (มีอยู่แล้ว — ปรับ undertone ให้ warm)
```css
.glass {
  background: rgba(22, 20, 18, 0.7);
  backdrop-filter: blur(12px);
  border: 1px solid rgba(245, 240, 235, 0.06);
}

.glass-strong {
  background: rgba(13, 11, 10, 0.85);
  backdrop-filter: blur(20px);
  border: 1px solid rgba(245, 240, 235, 0.08);
}
```

### Motion Rules
1. ใช้ animation เท่าที่จำเป็น — signature คือ minimal
2. `fade-in-up` สำหรับ element ตอน mount เท่านั้น
3. `glow-pulse` ใช้เฉพาะ hero — ไม่ใช้ทั้งหน้า
4. `shimmer` เฉพาะ skeleton
5. ห้าม animation infinite ที่ไม่จำเป็น (ยกเว้น audiowave ใน player)
6. ต้องรองรับ `prefers-reduced-motion` — ปิด animation ทั้งหมด

---

## 6. Spacing & Layout

### Breakpoints (Tailwind defaults)

| Breakpoint | Width | Usage |
|---|---|---|
| mobile (default) | < 640px | Single column, stacked |
| `sm` | ≥ 640px | Two columns possible |
| `md` | ≥ 768px | Sidebar + main layout |
| `lg` | ≥ 1024px | Full layout |
| `xl` | ≥ 1280px | Extra breathing room |

### Container & Section Spacing
```css
/* Content max-width */
max-w-7xl (1280px) — ใช้กับทุก layout container

/* Section padding */
px-4 py-8           → mobile
px-6 md:px-8 py-12  → tablet
px-8 lg:px-12       → desktop
```

### Gap System

| Context | Gap | ใช้กับ |
|---|---|---|
| `gap-2` | 8px | Badge + text, icon + label |
| `gap-3` | 12px | Small cards, setting rows |
| `gap-4` | 16px | Card grids, form fields |
| `gap-6` | 24px | Section spacing, major components |
| `gap-8` | 32px | Page sections, footer columns |

### Page Layout Pattern

```
┌──────────────────────────────────────────────┐
│  Navbar (fixed, h-16, glass-strong)          │
├──────────────────────────────────────────────┤
│  max-w-7xl mx-auto px-4 md:px-6              │
│  ┌────────────────────────────────────────┐  │
│  │  Hero / Page Header (pt-16 md:pt-24)   │  │
│  │  gap-6 ระหว่าง eyebrow + heading + desc │  │
│  ├────────────────────────────────────────┤  │
│  │  Main Content                          │  │
│  │  - 1 col mobile, 2-3 col desktop      │  │
│  │  - gap-4 ระหว่างการ์ด                  │  │
│  │  - gap-6 ระหว่าง section               │  │
│  ├────────────────────────────────────────┤  │
│  │  Footer                                │  │
│  └────────────────────────────────────────┘  │
└──────────────────────────────────────────────┘
```

### Specific Layout Rules
- **Upload page**: upload box กลางหน้าจอ, `max-w-xl`
- **Player/Mixer**: stem panels แนวนอน (1 col mobile, 2x2 grid desktop)
- **Account/Settings**: `max-w-2xl` centered, single column form
- **Pricing**: 3 cards (1 col mobile, 3 col md+), `gap-6`
- **API docs**: 2 col (sidebar nav + content)

---

## 7. ที่มาของข้อมูล (Source of Truth)

- Design นี้รวบรวมจาก pattern ที่มีอยู่จริงในโค้ด แล้วปรับโทนให้เป็น "Warm Studio"
- สี action/stem เดิมใน `AdvancedMultiTrackPlayer.tsx`, `HowItWorks.tsx`, `UserMenu.tsx` ยังคงใช้ได้ตามเดิม (ดู Section 2.3)
- ถ้าจะแก้ไข design system ให้อัปเดตไฟล์นี้ก่อนเสมอ แล้วค่อยแก้โค้ด
