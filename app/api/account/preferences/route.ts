import { NextResponse } from "next/server";
import { requireSession } from "@/lib/auth";
import { prisma } from "@/lib/prisma";

const VALID_THEMES = ["DARK", "LIGHT"];
const VALID_LANGUAGES = ["TH", "EN"];

export async function PUT(req: Request) {
  const { session, response: authResponse } = await requireSession();
  if (authResponse) return authResponse;

  const { theme, language, emailNotifications } = await req.json();

  if (theme !== undefined && !VALID_THEMES.includes(theme)) {
    return NextResponse.json({ error: "Invalid theme value" }, { status: 400 });
  }
  if (language !== undefined && !VALID_LANGUAGES.includes(language)) {
    return NextResponse.json({ error: "Invalid language value" }, { status: 400 });
  }
  if (emailNotifications !== undefined && typeof emailNotifications !== "boolean") {
    return NextResponse.json({ error: "emailNotifications must be a boolean" }, { status: 400 });
  }

  const user = await prisma.user.update({
    where: { id: session.user.id },
    data: {
      ...(theme !== undefined && { theme }),
      ...(language !== undefined && { language }),
      ...(emailNotifications !== undefined && { emailNotifications }),
    },
  });

  return NextResponse.json({
    preferences: {
      theme: user.theme,
      language: user.language,
      emailNotifications: user.emailNotifications,
    },
  });
}
