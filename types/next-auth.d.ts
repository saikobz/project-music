import { DefaultSession } from "next-auth";

declare module "next-auth" {
  interface Session {
    user: {
      id: string;
      tier: string;
      omiseCustomerId?: string;
    } & DefaultSession["user"];
  }

  interface JWT {
    id: string;
    tier: string;
    omiseCustomerId?: string;
  }
}
