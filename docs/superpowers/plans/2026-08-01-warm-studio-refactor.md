# Warm Studio — Color Refactoring Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Refactor all hardcoded neutral/glass colors in frontend (313 occurrences, 36 files) to match the "Warm Studio" DESIGN.md palette.

**Architecture:** Mechanical find-and-replace — swap old hex values → new warm hex values. No structural changes. Each task batch is verified with `npm run type-check && npm run lint`.

**Tech Stack:** Next.js 15, React 19, TypeScript, Tailwind CSS 4

## Global Constraints

- All neutral colors must use warm palette from `DESIGN.md`
- Action/stem/semantic colors (purple, cyan, amber, green, red, coral brand) stay unchanged
- Tailwind named colors (e.g. `text-purple-400`) stay unchanged
- Verify with `npm run type-check && npm run lint` after each task — never `npm run build` while dev server is running
- Commit after each task

---

## Color Mapping Reference

| Old Hex | New Hex | Occurrences |
|---|---|---|
| `#0A0A0A` | `#0D0B0A` | 59 |
| `#121212` | `#161412` | 26 |
| `#1A1A1A` | `#1E1B18` | 47 |
| `#2A2A2A` | `#2C2824` | 56 |
| `#333333`, `#333` | `#36322E` | 36 |
| `#F3F3F3` | `#F5F0EB` | 48 |
| `#A0A0A0` | `#A09890` | 16 |
| `#555555`, `#555` | `#5C5854` | 23 |
| glass rgba(20,20,20,...) | rgba(22,20,18,...) | 2 |
| glass rgba(14,14,14,...) | rgba(13,11,10,...) | 2 |

**Note:** `#333` is a shorthand — after replacement must become `#36322E` (not `#363` as that's a different color). Same with `#555` → `#5C5854`. Always replace `#333333` before `#333`, and `#555555` before `#555`.

**ReplaceAll order per file:**
1. `#333333` → `#36322E`
2. `#555555` → `#5C5854`
3. `#0A0A0A` → `#0D0B0A`
4. `#121212` → `#161412`
5. `#1A1A1A` → `#1E1B18`
6. `#2A2A2A` → `#2C2824`
7. `#F3F3F3` → `#F5F0EB`
8. `#A0A0A0` → `#A09890`
9. `#333` → `#36322E`
10. `#555` → `#5C5854`

---

### Task 1: Foundation — globals.css + layout.tsx

**Files:**
- Modify: `app/globals.css` (4 changes)
- Modify: `app/layout.tsx` (1 change)

**Changes:**

`app/globals.css` — update `.glass` and `.glass-strong` utilities:
```css
/* OLD */
.glass {
  background: rgba(20, 20, 20, 0.7);
  backdrop-filter: blur(12px);
  border: 1px solid rgba(255, 255, 255, 0.06);
}
.glass-strong {
  background: rgba(14, 14, 14, 0.85);
  backdrop-filter: blur(20px);
  border: 1px solid rgba(255, 255, 255, 0.08);
}
```
```css
/* NEW */
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

`app/layout.tsx:18`:
```
// OLD: bg-[#0A0A0A] text-[#F3F3F3]
// NEW: bg-[#0D0B0A] text-[#F5F0EB]
```

- [ ] Step 1: Apply all 5 changes using Edit tool
- [ ] Step 2: Run `npm run type-check` — expect PASS
- [ ] Step 3: Run `npm run lint` — expect PASS
- [ ] Step 4: Commit `refactor: warm palette — foundation (globals.css + layout.tsx)`

---

### Task 2: Core Components (14 files)

**Files — batch 2a (simple):**
- Modify: `app/components/Navbar.tsx`
- Modify: `app/components/Footer.tsx`
- Modify: `app/components/HowItWorks.tsx`
- Modify: `app/components/FaqSection.tsx`
- Modify: `app/components/UserMenu.tsx`
- Modify: `app/components/AudioAnalysis.tsx`
- Modify: `app/components/ExportMasterModal.tsx`
- Modify: `app/components/SingleExportModal.tsx`
- Modify: `app/components/settings/AutoEqSettings.tsx`
- Modify: `app/components/settings/CompressorSettings.tsx`
- Modify: `app/components/settings/PitchShiftSettings.tsx`

**Files — batch 2b (complex, has JS string colors):**
- Modify: `app/components/WaveformPlayer.tsx`
- Modify: `app/components/AdvancedMultiTrackPlayer.tsx`
- Modify: `app/components/UploadBox.tsx` (largest file, most changes)

**Mapping for all components:** Same 8 color swaps as above. The component files only use Tailwind arbitrary values — find `bg-[#0A0A0A]` → `bg-[#0D0B0A]`, etc.

**Special attention in WaveformPlayer.tsx:**
- Line 25: `waveColor: "#333333"` → `waveColor: "#36322E"` (JS string, not className)
- Line 158: `` `linear-gradient(to right, #E5A93D ${volume}%, #333 ${volume}%)` `` → change `#333` to `#36322E`

**Special attention in AdvancedMultiTrackPlayer.tsx:**
- `STEM_THEME` object (lines 18-43): uses `from-[#2A1321]`, `from-[#2B1E0D]`, `to-[#140C19]` etc. — stem-specific tints, **DO NOT CHANGE**

**Special attention in UploadBox.tsx:**
- Largest file (~662 lines) with ~45 occurrences — use replaceAll per color. Do NOT touch the 4 action color hexes (`#A78BFA`, `#22D3EE`, `#E5A93D`, `#34D399`).

- [ ] Step 1: Apply all batch 2a changes using Edit tool
- [ ] Step 2: Apply batch 2b changes (WaveformPlayer, AdvancedMultiTrackPlayer, UploadBox) with extra care for JS string constants
- [ ] Step 3: Run `npm run type-check` — expect PASS
- [ ] Step 4: Run `npm run lint` — expect PASS
- [ ] Step 5: Commit `refactor: warm palette — all components`

---

### Task 3: Route Pages (10 files)

**Files:**
- Modify: `app/page.tsx`
- Modify: `app/error.tsx`
- Modify: `app/studio/page.tsx`
- Modify: `app/pricing/page.tsx`
- Modify: `app/about/page.tsx`
- Modify: `app/guide/page.tsx`
- Modify: `app/support/page.tsx`
- Modify: `app/api-pricing/page.tsx`
- Modify: `app/privacy/page.tsx`
- Modify: `app/terms/page.tsx`

**Mapping:** Same 8 color swaps.

- [ ] Step 1: Apply all changes to route pages
- [ ] Step 2: Run `npm run type-check` — expect PASS
- [ ] Step 3: Run `npm run lint` — expect PASS
- [ ] Step 4: Commit `refactor: warm palette — all route pages`

---

### Task 4: Auth + Dashboard (2 files)

**Files:**
- Modify: `app/auth/signin/page.tsx`
- Modify: `app/dashboard/history/page.tsx`

**Mapping:** Same 8 color swaps.

- [ ] Step 1: Apply all changes
- [ ] Step 2: Run `npm run type-check` — expect PASS
- [ ] Step 3: Run `npm run lint` — expect PASS
- [ ] Step 4: Commit `refactor: warm palette — auth + dashboard pages`

---

### Task 5: Account Pages (8 files)

**Files:**
- Modify: `app/account/page.tsx`
- Modify: `app/account/ProfileSection.tsx`
- Modify: `app/account/PasswordSection.tsx`
- Modify: `app/account/ConnectedAccountsSection.tsx`
- Modify: `app/account/PreferencesSection.tsx`
- Modify: `app/account/BillingSection.tsx`
- Modify: `app/account/DataSection.tsx`
- Modify: `app/account/confirm-delete/page.tsx`

**Mapping:** Same 8 color swaps.

- [ ] Step 1: Apply all changes
- [ ] Step 2: Run `npm run type-check` — expect PASS
- [ ] Step 3: Run `npm run lint` — expect PASS
- [ ] Step 4: Commit `refactor: warm palette — account pages`

---

### Task 6: Final Verification

- [ ] Step 1: Run `npm run type-check` — confirm PASS on clean working tree
- [ ] Step 2: Run `npm run lint` — confirm PASS
- [ ] Step 3: Run `npx jest` — confirm all frontend tests pass (no color-dependent tests expected)
- [ ] Step 4: Final grep to verify no old colors remain: `rg '#0A0A0A|#121212|#1A1A1A|#2A2A2A|#333333|#F3F3F3|#A0A0A0|#555555' --include='*.tsx' --include='*.css' app/` — expect ZERO matches for the target old colors (except `#333`/`#555` shorthands which must be checked separately)
- [ ] Step 5: Commit (if needed) `chore: verify warm palette refactoring complete`
