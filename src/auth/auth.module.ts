import { Module } from '@nestjs/common';
import { AuthController } from './auth.controller';
import { AuthService } from './auth.service';
import { PrismaService } from 'src/prisma/prisma.service';
import { JwtModule } from '@nestjs/jwt';
import { jwtConstants } from './constants';
import { APP_GUARD } from '@nestjs/core';
import { AuthGuard } from './auth.guard';
//import { APP_GUARD } from '@nestjs/core';
@Module({
    imports:[
      JwtModule.register(
        {
          global:true,
          secret:jwtConstants.secret,
          signOptions:{
            expiresIn:'6hr'
          }
        }
      )
    ],
    controllers: [AuthController],
    providers: [
      AuthService, 
      PrismaService,
      {
        provide: APP_GUARD,
        useClass: AuthGuard,
      }
     ],
    exports:[AuthService]
})
export class AuthModule {}
