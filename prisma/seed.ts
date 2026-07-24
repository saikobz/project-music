import { PrismaClient } from "@prisma/client";
import bcrypt from "bcryptjs";

const prisma = new PrismaClient();

async function main() {
  const email = "admin@harmoniq.ai";
  const password = "adminpassword123";
  const hashedPassword = await bcrypt.hash(password, 10);

  const user = await prisma.user.upsert({
    where: { email },
    update: {
      password: hashedPassword,
      name: "Admin HarmoniQ",
    },
    create: {
      email,
      name: "Admin HarmoniQ",
      password: hashedPassword,
      subscription: {
        create: {
          tier: "PRO",
          status: "ACTIVE",
        },
      },
    },
  });

  await prisma.subscription.upsert({
    where: { userId: user.id },
    update: { tier: "PRO", status: "ACTIVE" },
    create: {
      userId: user.id,
      tier: "PRO",
      status: "ACTIVE",
    },
  });

  console.log("Admin account created successfully!");
  console.log(`Email: ${email}`);
  console.log(`Password: ${password}`);
  console.log(`Tier: PRO (Unlimited Access)`);
}

main()
  .catch((e) => {
    console.error(e);
    process.exit(1);
  })
  .finally(async () => {
    await prisma.$disconnect();
  });
