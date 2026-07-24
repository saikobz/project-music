import bcrypt from "bcryptjs";
import { prisma } from "../lib/prisma";

describe("Credentials Auth Schema Test", () => {
  afterAll(async () => {
    await prisma.$disconnect();
  });

  it("should create user with hashed password", async () => {
    const hashedPassword = await bcrypt.hash("password123", 10);
    const user = await prisma.user.create({
      data: {
        email: `cred-${Date.now()}@example.com`,
        name: "Cred User",
        password: hashedPassword,
      },
    });

    expect(user.password).toBeDefined();
    const isMatch = await bcrypt.compare("password123", user.password!);
    expect(isMatch).toBe(true);

    await prisma.user.delete({ where: { id: user.id } });
  });
});
