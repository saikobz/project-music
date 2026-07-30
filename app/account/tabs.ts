export const ACCOUNT_TABS = [
  { id: "profile", label: "Profile", icon: "User" },
  { id: "password", label: "Password", icon: "KeyRound" },
  { id: "accounts", label: "Connected Accounts", icon: "Link" },
  { id: "preferences", label: "Preferences", icon: "Settings" },
] as const;

export type AccountTabId = (typeof ACCOUNT_TABS)[number]["id"];
