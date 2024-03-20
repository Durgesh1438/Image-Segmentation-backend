import { Injectable } from '@nestjs/common';
import Razorpay  =require('razorpay');
import { PrismaService } from 'src/prisma/prisma.service';


@Injectable()
export class RazorpayService {
    private razorpay: any;
    constructor(private  prismaService:PrismaService) {
        this.razorpay = new Razorpay({
          key_id: process.env.KEY_ID, // Replace with your Razorpay key ID
          key_secret: process.env.KEY_SECRET, // Replace with your Razorpay key secret
        });
      }
    
    
      async createOrder(amount: number): Promise<any> {
        const options = {
          amount: amount * 100, // Razorpay expects amount in paisa
          currency: 'INR', // Change it according to your currency
        };
    
        return await this.razorpay.orders.create(options);
      }
    
      async capturePayment(paymentId: string, amount: number): Promise<any> {
       try{
        const captureResponse=await this.razorpay.payments.capture(paymentId, amount * 100);
        console.log(captureResponse)
    
        return captureResponse;
       }
       catch(error){
        console.log(error)
        if (error.statusCode===400 &&  error.error.code === 'BAD_REQUEST_ERROR' && error.error.description === 'This payment has already been captured') {
            // Return a success response indicating that the payment has already been captured
            return { success: true, message: 'Payment has already been captured' };
          } else {
            console.log("Error capturing payment:", error);
            return {success:false,message:"Payment has failed"}
          }
       }
      }

    async freeTrail(email:string,startDate:Date,endDate:Date,freetrail:boolean,subscriptionPlan:string,picture:string){
        //console.log(email,startDate,endDate,freetrail)
        console.log(email,picture)
        try{
            const user=await this.prismaService.user.findUnique(
                {
                    where:{
                        email:email
                    }
                }
            )

            if(user.freetrail){
                return {
                    success:false,
                    message:"Already user have used their free trail!Please subscribe to continue"
                }
            }
            if(user){
                await this.prismaService.user.update({
                    where:{
                        email:email
                    },
                    data:{
                        startDate:startDate,
                        endDate:endDate,
                        freetrail:freetrail,
                        subscriptionPlan:subscriptionPlan
                    }
                }
                )
            }

            return {
                success:true,
                picture:picture,
                message:"Free Trail started"
            }
        }
        catch(error){
            return {
                success:false,
                message:"Free Trail is not initiated!Try Again!"
            }
        }
    }

    getprofile(data:any){
        return data;
    }

    async dbUpdate(isSubscriber:boolean,subscriptionPlan:string,startDate:Date,endDate:Date,email:string,picture:string):Promise<any>{
        console.log(isSubscriber,subscriptionPlan,startDate,endDate,email)
        try {
            // Find user by email
            const user = await this.prismaService.user.findUnique({
                where: {
                    email: email,
                },
            });

            if (user) {
                // Update user's subscription details
                await this.prismaService.user.update({
                    where: {
                        email: email,
                    },
                    data: {
                        isSubscriber: isSubscriber,
                        subscriptionPlan: subscriptionPlan,
                        startDate: startDate,
                        endDate: endDate,
                    },
                });
            } 
            
            return {
                success:true,
                picture:picture,
                message:"subscribed successfully"
            }

        } catch (error) {
            return {
                success:false,
                message:"subscription failed login in to access"
            }
        }
    }
}
