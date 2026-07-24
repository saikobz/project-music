jest.mock("@auth/prisma-adapter", () => ({
  PrismaAdapter: jest.fn(),
}));

import { authOptions } from "../app/api/auth/[...nextauth]/route";

describe("Multi-Auth Providers Test", () => {
  it("should configure Credentials, Google, Facebook, and LINE providers", () => {
    const providerIds = authOptions.providers.map((p) => p.id);
    expect(providerIds).toContain("credentials");
    expect(providerIds).toContain("google");
    expect(providerIds).toContain("facebook");
    expect(providerIds).toContain("line");
  });
});
