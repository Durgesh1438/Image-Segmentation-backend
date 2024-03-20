/* eslint-disable @typescript-eslint/no-unused-vars */
import { Body, Controller, Post,SetMetadata,HttpCode,HttpStatus, Request, Get, UseGuards} from '@nestjs/common';
import { AuthService } from './auth.service';
import { AuthGuard } from './auth.guard';

export const IS_PUBLIC_KEY = 'isPublic';
export const Public = () => SetMetadata(IS_PUBLIC_KEY, true);

@Controller('auth')
export class AuthController {
  constructor(private readonly authService: AuthService) {}

  @Public()
  @HttpCode(HttpStatus.OK)
  @Post("login")
  async signIn(@Body() signInDto: Record<string, any>) {
     return  await this.authService.signIn(signInDto);
  }
  
  @Public()
  @HttpCode(HttpStatus.OK)
  @Post('google')
  async signUpGoogle(@Body() signUpDto: Record<string, any>) {
    return this.authService.signInWithGoogle(signUpDto);
  }

 
  @Public()
  @HttpCode(HttpStatus.OK)
  @Post('signup')
  async signUp(@Body() signUpDto: Record<string, any>) {
    return this.authService.signUp(signUpDto);
  }

  
  @Get('profile')
  getProfile(@Request() req) {
    return req.user;
  }
}
