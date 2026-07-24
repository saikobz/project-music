import { prisma } from "../lib/prisma";

describe("Database Schema Test", () => {
  afterAll(async () => {
    await prisma.$disconnect();
  });

  it("should create user with default FREE subscription tier", async () => {
    const user = await prisma.user.create({
      data: {
        email: `test-${Date.now()}@example.com`,
        name: "Test User",
        subscription: {
          create: {
            tier: "FREE",
            status: "ACTIVE",
          },
        },
      },
      include: { subscription: true },
    });

    expect(user.email).toContain("@example.com");
    expect(user.subscription?.tier).toBe("FREE");

    // Clean up
    await prisma.user.delete({ where: { id: user.id } });
  });
});
