jest.mock("@next-auth/prisma-adapter", () => ({
  PrismaAdapter: jest.fn(),
}));

import { authOptions } from "../lib/auth";

describe("Multi-Auth Providers Test", () => {
  it("should configure Credentials, Google, Facebook, and LINE providers", () => {
    const providerIds = authOptions.providers.map((p) => p.id);
    expect(providerIds).toContain("credentials");
    expect(providerIds).toContain("google");
    expect(providerIds).toContain("facebook");
    expect(providerIds).toContain("line");
  });
});

describe("LINE Provider Profile Fallback", () => {
  const getLineProfile = () => {
    const lineProvider = authOptions.providers.find((p) => p.id === "line") as {
      options: { profile?: (profile: Record<string, any>) => any };
    };
    expect(lineProvider).toBeDefined();
    expect(lineProvider.options?.profile).toBeDefined();
    return lineProvider.options.profile!;
  };

  it("should generate a stable fallback email when LINE does not provide one", () => {
    const profileFn = getLineProfile();

    const profile = profileFn({
      sub: "user-123",
      name: "Padon",
      picture: "https://profile.line-scdn.net/pic",
    });

    expect(profile.email).toBe("line_user-123@line.local");
    expect(profile.id).toBe("user-123");
    expect(profile.name).toBe("Padon");
  });

  it("should use the real email when LINE provides one", () => {
    const profileFn = getLineProfile();

    const profile = profileFn({
      sub: "user-123",
      name: "Padon",
      email: "padon@example.com",
      picture: "https://profile.line-scdn.net/pic",
    });

    expect(profile.email).toBe("padon@example.com");
  });
});
