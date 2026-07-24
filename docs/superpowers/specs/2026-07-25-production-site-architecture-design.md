# Production Site Architecture & Page Structure Design Spec

**Date:** 2026-07-25  
**Status:** Approved  
**Project:** HarmoniQ (Music Separator & EQ/Compressor)

---

## 1. Executive Summary & Goals

HarmoniQ is a web-based AI music stem separation and mastering platform built for music creators, producers, and audio engineers. To scale to a production-grade commercial SaaS, the application requires a clear separation between marketing/onboarding pages, a dedicated distraction-free studio workspace, user project history, billing/subscription management, and legal compliance pages (required for payment gateway verification such as Omise).

### Key Goals
- **Architectural Separation:** Separate the public Marketing Landing Page (`/`) from the core DAW-style Studio Workspace (`/studio`).
- **Comprehensive Page Hierarchy:** Establish a complete B2C SaaS page map including Studio, History, Pricing, Account, Guide, About, Terms, Privacy, Support, and Auth.
- **Enhanced User Retention & UX:** Provide project history (`/dashboard/history`) for users to re-download processed stems without re-processing.
- **Compliance & Gateway Approval:** Include legally compliant Terms of Service and Privacy Policy pages for Omise payment gateway onboarding.

---

## 2. Page Hierarchy & Next.js App Router Mapping

| Route | Page Name | Access Level | Description & Core Components |
| :--- | :--- | :---: | :--- |
| `/` | **Landing Page** | Public | High-converting marketing hero, interactive stem demo audio player, feature highlights (AI Models, AutoEQ, Compressor), pricing teaser, testimonials, CTA to Studio. |
| `/studio` | **Studio Workspace** | Public / Auth | Fullscreen DAW-style workspace. Contains `UploadBox`, WaveSurfer.js multi-track player, track controls (Mute, Solo, Volume, Pitch Shift, AutoEQ, Compressor), and Export Modals. |
| `/dashboard/history` | **Project History** | Authenticated | List of previously uploaded & processed songs and stems. Audio preview player, download stems button, delete project button. |
| `/pricing` | **Pricing & Plans** | Public | Tier comparison matrix (Free, Basic 99 THB/mo, Pro 299 THB/mo), FAQ accordion, integrated `CheckoutModal` (Omise Card/PromptPay). |
| `/account` | **Account & Billing** | Authenticated | Profile info, subscription status, monthly quota usage bar (`X / Y songs used`), plan management (Upgrade/Cancel), billing receipts. |
| `/about` | **About & AI Models** | Public | Product background, explanation of AI models (Open-Unmix, LSTM vs CNN AutoEQ). |
| `/guide` | **User Guide & Docs** | Public | Step-by-step tutorials, Audio mastering LUFS guidelines, supported WAV file formats, troubleshooting. |
| `/terms` | **Terms of Service** | Public | Audio ownership rights, user code of conduct, liability disclaimer, service terms. |
| `/privacy` | **Privacy Policy** | Public | PDPA compliance, cookie policy, temporary audio file auto-cleanup schedule (24h retention). |
| `/support` | **Help & Support** | Public | Support contact form, issue report ticket, system status link. |
| `/auth/signin` | **Sign In** | Public | NextAuth.js login page (Google OAuth + Email Magic Link). |

---

## 3. Migration & Refactoring Strategy

### 3.1 Moving Current Home Page to `/studio`
- Currently, `app/page.tsx` renders `UploadBox.tsx` directly.
- **Action:** Move the Studio logic (`UploadBox`) to `app/studio/page.tsx`.
- **Action:** Re-purpose `app/page.tsx` to be the marketing landing page with visual hero, interactive audio player demo, and direct CTAs pointing to `/studio`.

### 3.2 Navigation & Layout Consistency
- **Navbar (`app/components/Navbar.tsx`):**
  - Updated with links: `Studio`, `Pricing`, `Guide`, `History` (if logged in), and `Account / Sign In`.
- **Footer (`app/components/Footer.tsx`):**
  - Group links into Product (`Studio`, `Pricing`), Resources (`Guide`, `About`), Legal (`Terms`, `Privacy`), and Support (`Help`).

---

## 4. Auth & Middleware Protection

```typescript
// Middleware Route Protection Rule
const protectedRoutes = ['/dashboard/history', '/account'];
const studioQuotaGuard = ['/studio']; // Guests can use free quota, logged in users use tier quota
```

---

## 5. Verification & Quality Assurance Plan

1. **Routing Verification:** Test navigation across all 11 routes using `npm run type-check` and browser routing.
2. **Responsive Design:** Ensure both Landing Page and Studio Workspace adjust seamlessly between Desktop, Tablet, and Mobile views.
3. **Omise & Compliance Audit:** Verify that `/terms` and `/privacy` links are visible on the footer and inside the `/pricing` checkout flow.
