
import { MiddlewareConsumer, Module } from '@nestjs/common';
import { AppController } from './app.controller';
import { AppService } from './app.service';
import { ServeStaticModule } from '@nestjs/serve-static';
import { join } from 'path';
import { AuthMiddleware } from './auth/auth.middleware';
import { PrismaService } from './prisma/prisma.service';
import { AuthModule } from './auth/auth.module';
import { MulterModule } from '@nestjs/platform-express';
import { RazorpayModule } from './razorpay/razorpay.module';
import { ConfigModule } from '@nestjs/config';
import { AdminModule } from './admin/admin.module';
@Module({
  imports: [
    ServeStaticModule.forRoot({
      rootPath: join(__dirname, '..', 'storage'), // Change the path as per your setup
      serveRoot:'/storage',
      
      
    }),
    ConfigModule.forRoot({
      isGlobal:true
    }),
    MulterModule.register(),
    AuthModule,
    RazorpayModule,
    AdminModule,
  ],
  controllers: [AppController,],
  providers: [AppService,  PrismaService,],
})
export class AppModule {
  configure(consumer: MiddlewareConsumer) {
    consumer
      .apply(AuthMiddleware) // Apply the AuthMiddleware to all routes
      .forRoutes('*');
  }
}
