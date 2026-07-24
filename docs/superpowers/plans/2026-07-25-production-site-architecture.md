# Production Site Architecture Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Transform HarmoniQ into a production-ready B2C web app by separating the Marketing Landing Page (`/`) from the Studio Workspace (`/studio`), adding Legal Pages (`/terms`, `/privacy`), Project History (`/dashboard/history`), and updated Navigation & Footer.

**Architecture:** Split existing `app/page.tsx` studio interface into `app/studio/page.tsx`. Turn `app/page.tsx` into a high-converting Marketing Landing Page. Add static legal & compliance pages for Omise onboarding, and create a user track history page under `/dashboard/history`.

**Tech Stack:** Next.js 15 (App Router), React 19, TypeScript, Tailwind CSS 4, Lucide React icons.

## Global Constraints

- Python 3.10 compatibility for backend endpoints.
- Tailwind CSS 4 for all styling (no arbitrary inline styles unless dynamic).
- Use `npm run type-check` (`npx tsc --noEmit`) for verification instead of `npm run build` during active dev server.
- All file paths must be absolute or relative to project root.

---

### Task 1: Refactor Studio Workspace & Marketing Landing Page (`/studio` and `/`)

**Files:**
- Create: `app/studio/page.tsx`
- Modify: `app/page.tsx`
- Modify: `app/components/Navbar.tsx`

**Interfaces:**
- Consumes: `UploadBox.tsx`, `Navbar.tsx`, `Footer.tsx`
- Produces: `/studio` route for workspace, `/` route for marketing landing page

- [ ] **Step 1: Create `/studio` route page**

Write `app/studio/page.tsx`:
```tsx
"use client";

import React, { useState } from "react";
import { Navbar } from "../components/Navbar";
import { Footer } from "../components/Footer";
import UploadBox from "../components/UploadBox";

export default function StudioPage() {
  const [isExpanded, setIsExpanded] = useState(false);

  return (
    <div className={`flex min-h-screen flex-col bg-[#0A0A0A] text-[#F3F3F3] transition-all duration-300 ${
      isExpanded ? "overflow-y-auto" : "md:h-screen md:overflow-hidden overflow-y-auto"
    }`}>
      <Navbar />

      <main className={`mx-auto flex-grow w-full px-4 py-4 md:py-6 space-y-4 md:space-y-6 flex flex-col justify-center transition-all duration-300 ${
        isExpanded ? "max-w-7xl" : "max-w-5xl"
      }`}>
        <header className="flex flex-col gap-2">
          <h1 className="text-3xl md:text-4xl font-bold leading-tight">
            HarmoniQ AI Studio Workspace
          </h1>
          <p className="text-sm md:text-base text-[#8E8E8E] max-w-3xl font-light">
            อัปโหลดไฟล์ WAV เพื่อแยกเสียงดนตรีด้วย AI พร้อม EQ, Compressor, Pitch Shift และการวิเคราะห์ Tempo/Key
          </p>
        </header>

        <section className="bg-[#121212] border border-[#2A2A2A] rounded-2xl shadow-2xl overflow-hidden">
          <UploadBox onHeightChange={setIsExpanded} />
        </section>
      </main>

      <Footer />
    </div>
  );
}
```

- [ ] **Step 2: Refactor `app/page.tsx` into Marketing Landing Page**

Write `app/page.tsx` with Hero, Feature Highlights, Interactive Audio Teaser, and CTAs.

- [ ] **Step 3: Run Type Check**

Run: `npx tsc --noEmit`
Expected: PASS with 0 errors

- [ ] **Step 4: Commit**

```bash
git add app/studio/page.tsx app/page.tsx
git commit -m "feat: separate studio workspace to /studio and add marketing landing page to /"
```

---

### Task 2: Create Legal & Compliance Pages (`/terms`, `/privacy`, `/support`)

**Files:**
- Create: `app/terms/page.tsx`
- Create: `app/privacy/page.tsx`
- Create: `app/support/page.tsx`
- Modify: `app/components/Footer.tsx`

**Interfaces:**
- Consumes: `Navbar.tsx`, `Footer.tsx`
- Produces: Compliance routes `/terms`, `/privacy`, `/support` required for Omise onboarding

- [ ] **Step 1: Create `app/terms/page.tsx`**

Implement Terms of Service page covering audio ownership rights, user limits, and disclaimers.

- [ ] **Step 2: Create `app/privacy/page.tsx`**

Implement Privacy Policy page detailing PDPA compliance and 24h temporary file cleanup policy.

- [ ] **Step 3: Create `app/support/page.tsx`**

Implement Support page with contact info and feedback form.

- [ ] **Step 4: Update `app/components/Footer.tsx`**

Add links to Terms, Privacy, Support, Studio, Pricing, About, and Guide.

- [ ] **Step 5: Run Type Check**

Run: `npx tsc --noEmit`
Expected: PASS

- [ ] **Step 6: Commit**

```bash
git add app/terms/page.tsx app/privacy/page.tsx app/support/page.tsx app/components/Footer.tsx
git commit -m "feat: add terms of service, privacy policy, and support pages"
```

---

### Task 3: Create Project History Page (`/dashboard/history`)

**Files:**
- Create: `app/dashboard/history/page.tsx`

**Interfaces:**
- Consumes: NextAuth session, Navbar, Footer
- Produces: Project history table with audio playback preview & stem download links

- [ ] **Step 1: Create `app/dashboard/history/page.tsx`**

Implement history dashboard showing processed WAV stems, timestamps, and download actions.

- [ ] **Step 2: Run Type Check**

Run: `npx tsc --noEmit`
Expected: PASS

- [ ] **Step 3: Commit**

```bash
git add app/dashboard/history/page.tsx
git commit -m "feat: add project history dashboard page"
```

---

### Task 4: Enhance Navigation Bar (`app/components/Navbar.tsx`)

**Files:**
- Modify: `app/components/Navbar.tsx`

- [ ] **Step 1: Update Navbar links**

Add active path checking and links for `Studio`, `Pricing`, `Guide`, `History`, `Account / Sign In`.

- [ ] **Step 2: Verify and Commit**

Run: `npx tsc --noEmit`
```bash
git add app/components/Navbar.tsx
git commit -m "feat: update navbar with studio and history navigation links"
```
