import { Injectable, NestMiddleware, UnauthorizedException } from '@nestjs/common';
import { JwtService } from '@nestjs/jwt';
import { Request, Response, NextFunction } from 'express';
import { jwtConstants } from './constants';
@Injectable()
export class AuthMiddleware implements NestMiddleware {
  constructor(private readonly jwtService: JwtService) {}

  async use(req: Request, res: Response, next: NextFunction) {
    //console.log(req)
    res.setHeader('Cross-Origin-Opener-Policy', 'same-origin');
    const isExcludedRoute = req.originalUrl === '/auth/login' || req.originalUrl === '/auth/signup'|| req.originalUrl==='/auth/google'; // Define excluded routes

    if (isExcludedRoute) {
      return next(); // Bypass authentication for excluded routes
    }
    const token = this.extractTokenFromHeader(req);
    if (!token) {
      throw new UnauthorizedException('No token provided');
    }

    try {
      const payload = await this.jwtService.verifyAsync(token, { secret: jwtConstants.secret  });
      req['user'] = payload; // Attach the decoded user information to the request object
      next(); // Pass the request to the next middleware
    } catch (error) {
      throw new UnauthorizedException('Failed to authenticate token');
    }
  }

  private extractTokenFromHeader(req: Request): string | undefined {
    const authHeader = req.headers['authorization'];
    if (typeof authHeader !== 'undefined') {
      const [type, token] = authHeader.split(' ');
      if (type === 'Bearer' && token) {
        return token;
      }
    }
    return undefined;
  }
}
