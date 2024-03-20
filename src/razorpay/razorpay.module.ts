import { Module } from '@nestjs/common';
import { RazorpayController } from './razorpay.controller';
import { RazorpayService } from './razorpay.service';
import { ConfigModule } from '@nestjs/config';
import { PrismaService } from 'src/prisma/prisma.service';
@Module({
  imports:[
    ConfigModule.forRoot({
      isGlobal:true
    }),
  ],
  controllers: [RazorpayController],
  providers: [RazorpayService,PrismaService]
})
export class RazorpayModule {}
