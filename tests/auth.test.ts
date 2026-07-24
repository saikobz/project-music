jest.mock("@auth/prisma-adapter", () => ({
  PrismaAdapter: jest.fn(),
}));

import { authOptions } from "../app/api/auth/[...nextauth]/route";

describe("NextAuth Session Callback", () => {
  it("should enrich session with user tier and subscription status", async () => {
    const mockSession = { user: { email: "test@example.com", id: "u-1", tier: "FREE" }, expires: "2099-01-01" };
    const mockToken = { sub: "user-123", id: "user-123", tier: "PRO", omiseCustomerId: "cust_123" };

    if (authOptions.callbacks?.session) {
      const session = await authOptions.callbacks.session({
        session: mockSession as any,
        token: mockToken as any,
        user: {} as any,
        newSession: {} as any,
        trigger: "update",
      });

      const user = session?.user as any;
      expect(user?.tier).toBe("PRO");
      expect(user?.omiseCustomerId).toBe("cust_123");
    }
  });
});
