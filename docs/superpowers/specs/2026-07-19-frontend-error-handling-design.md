# Frontend Error Handling & UX Design Spec

## 1. Overview
The HarmoniQ web application currently lacks a robust, user-facing error handling mechanism on the frontend. When API calls fail or unexpected React rendering errors occur, the user experience degrades (e.g., stuck loading states, white screens). This specification details the implementation of a comprehensive error handling system using global Next.js Error Boundaries and Toast notifications.

## 2. Architecture & Components

### 2.1 Toast Notifications (`sonner`)
- **Library**: `sonner` will be used for its premium, modern aesthetics and lightweight nature.
- **Integration**: 
  - Add `<Toaster />` component to the root `app/layout.tsx`.
  - Configure the toaster to support both Light and Dark themes to match the application's aesthetic.
  - Set the default position (e.g., `bottom-right` or `top-center`).

### 2.2 Global Error Boundary (`app/error.tsx`)
- **Implementation**: Create a Next.js `error.tsx` file at the root `app/` directory.
- **UI/UX**: 
  - The fallback UI must align with the premium aesthetics of HarmoniQ (e.g., using glassmorphism, appropriate typography, and colors).
  - Include a clear, non-technical error message.
  - Provide a "Try Again" (Refresh) button that invokes the Next.js `reset()` function to attempt recovery without a full page reload.

## 3. Data Flow & Component Updates

Existing components that perform asynchronous operations (API calls) must be updated to handle errors gracefully:

1. **UploadBox (`app/components/UploadBox.tsx`)**:
   - Wrap the file upload Axios/Fetch calls in `try...catch` blocks.
   - On error: Trigger `toast.error(errorMessage)` and reset the loading/uploading state.
2. **Export Modals (`app/components/SingleExportModal.tsx`, `app/components/ExportMasterModal.tsx`)**:
   - Catch errors during the export API calls.
   - On error: Trigger `toast.error(errorMessage)` and reset the processing state.
3. **AdvancedMultiTrackPlayer / WaveformPlayer**:
   - Ensure that any errors during audio loading or processing trigger a toast notification rather than silently failing or breaking the player UI.

## 4. Testing Strategy
- **Manual API Failure**: Introduce an artificial error in the backend or upload an unsupported file type to verify the `sonner` toast notification appears and the UI returns to a usable state.
- **React Render Error**: Temporarily throw an explicit `new Error()` inside a deeply nested component to ensure the `app/error.tsx` boundary catches it and displays the fallback UI correctly.
