/* eslint-disable @typescript-eslint/no-unused-vars */
import {
  HttpException,
  Injectable,
  UnauthorizedException,
} from '@nestjs/common';
import { PrismaService } from 'src/prisma/prisma.service';
import { JwtService } from '@nestjs/jwt';
import { OAuth2Client } from 'google-auth-library';
import { googleclientId } from 'src/common/helper/uploads.helper';
@Injectable()
export class AuthService {
  private readonly googleClient: OAuth2Client;
  constructor(
    private prismaService: PrismaService,
    private jwtService: JwtService,
  ) {
    this.googleClient = new OAuth2Client(googleclientId);
  }

  async signIn(data: any) {
    const user = await this.prismaService.user.findFirst({
      where: { username: data?.username?.toString() },
    });
    console.log(user.password);

    if (user?.password !== data?.password) {
      throw new UnauthorizedException();
    }

    /*return {
            data : 'user logged in successfully'
        }*/
    const payload = { ...user, sub: user.username, username: user.username };
    console.log(payload);
    return {
      access_token: await this.jwtService.signAsync(payload),
      isSubscriber: user.isSubscriber,
      endDate:user.endDate,
      freetrail:user.freetrail,
      isAdmin:user.isAdmin
    };
  }

  async signUp(data: any) {
    const user = await this.prismaService.user.findFirst({
      where: { username: data?.username },
    });
    if (user) {
      throw new HttpException('Username already exists', 500);
    } else {
      const user = await this.prismaService.user.create({
        data,
      });
      return {
        data: user,
        message: 'User created successfully',
      };
    }
  }

  async signInWithGoogle(data: any) {
    try {
      const ticket = await this.googleClient.verifyIdToken({
        idToken: data.access_token,
        audience: googleclientId,
      });
      const payload = ticket.getPayload();
      console.log(payload);
      const userEmail = payload.email;
      let user = await this.prismaService.user.findFirst({
        where: { email: userEmail },
      });
      
      if (!user) {
        // If the user doesn't exist, create a new user record
        const givenName = payload.given_name.replace(/\s+/g, ''); // Remove spaces from given name
        const familyName = payload.family_name ? payload.family_name.replace(/\s+/g, '') : ''; // 
        const username = `${givenName}${familyName}`;
        user = await this.prismaService.user.create({
           data :{
            username: String(username),
            email: String(userEmail),
            password:String(payload.given_name+payload.family_name)
           }
        });
      }
      console.log(user)
      const jwtPayload = { ...user, sub: user.username, username: user.username,picture:payload.picture };
        return {
            access_token: await this.jwtService.signAsync(jwtPayload),
            username:user.username,
            picture:payload.picture,
            isSubscriber:user.isSubscriber,
            endDate:user.endDate,
            freetrail:user.freetrail,
            isAdmin : user.isAdmin
        };
    } catch (error) {
      console.error('Google sign-in error:', error);
      throw new UnauthorizedException('Failed to sign in with Google');
    }
  }
}
