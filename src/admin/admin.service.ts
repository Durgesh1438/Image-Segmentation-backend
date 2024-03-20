import { Injectable } from '@nestjs/common';
import { PrismaService } from 'src/prisma/prisma.service';

@Injectable()
export class AdminService {
  constructor(private prismaService: PrismaService) {}

  async getAllUsers(): Promise<any> {
    const userData = await this.prismaService.user.findMany();
    console.log(userData);
    return userData;
  }

  async searchUser(email: string): Promise<any> {
    try {
      const user = await this.prismaService.user.findUnique({
        where: {
          email: email,
        },
      });
      if (user) {
        return {
          success:true,
          user
        };
      } else {
        return {
          success: false,
          message: 'user not found ! please type crct email address',
        };
      }
    } catch (error) {
      console.log('error finding user:', error);
    }
  }

  async updateUser(email: string, pack: string): Promise<any> {
    try {
      const user = await this.prismaService.user.findUnique({
        where: {
          email: email,
        },
      });

      if (!user) {
        return {
          success: false,
          message: 'user not found',
        };
      }

      const  startDate = new Date();
      const  endDate = new Date(startDate);
      switch (pack) {
        case '1 day':
          endDate.setDate(endDate.getDate() + 1);
          break;
        case '7 days':
          endDate.setDate(endDate.getDate() + 7);
          break;
        case '7 days trial':
          endDate.setDate(endDate.getDate() + 7);
          break;
        case '1 month':
          endDate.setMonth(endDate.getMonth() + 1);
          break;
        case '2 months':
          endDate.setMonth(endDate.getMonth() + 2);
          break;
        case '3 months':
          endDate.setMonth(endDate.getMonth() + 3);
          break;
        case '1 year':
          endDate.setFullYear(endDate.getFullYear() + 1);
          break;
        default :
          throw new Error('Unsupported subscription pack')
      }

      if (pack === '7 days trial') {
        await this.prismaService.user.update(
          {
            where:{
              email:email
            },
            data:{
              isSubscriber:true,
              freetrail:true,
              startDate:startDate,
              endDate:endDate,
              subscriptionPlan:pack
            }
          }
        )
      }
      else{
        await this.prismaService.user.update(
          {
            where:{
              email:email
            },
            data:{
              isSubscriber:true,
              startDate:startDate,
              endDate:endDate,
              subscriptionPlan:pack,
            }
          }
        )
      }

      return {
        success:true,
        message:"user subscription plan updated successfully"
      }
    } catch (error) {
       return {
        success:false,
        message:"user subscription plan denied!!Try Again!!"
       }
    }
  }
}
