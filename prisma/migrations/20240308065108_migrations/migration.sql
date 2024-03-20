-- AlterTable
ALTER TABLE "User" ADD COLUMN     "endDate" TIMESTAMP(3),
ADD COLUMN     "isSubscriber" BOOLEAN NOT NULL DEFAULT false,
ADD COLUMN     "startDate" TIMESTAMP(3),
ADD COLUMN     "subscriptionPlan" TEXT;
