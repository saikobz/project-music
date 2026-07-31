-- CreateIndex
CREATE INDEX "UsageQuota_userId_idx" ON "UsageQuota"("userId");

-- CreateIndex
CREATE UNIQUE INDEX "UsageQuota_userId_periodStart_key" ON "UsageQuota"("userId", "periodStart");
