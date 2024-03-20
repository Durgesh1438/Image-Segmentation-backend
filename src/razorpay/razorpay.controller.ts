/* eslint-disable @typescript-eslint/no-unused-vars */
import { Body, Controller, Get, Post, Request } from '@nestjs/common';
import { RazorpayService } from './razorpay.service';
@Controller('razorpay')
export class RazorpayController {
  constructor(private readonly razorpayService: RazorpayService) {}
  @Post('/subscribe')
  getHello(@Request() req, @Body() body) {
    return this.razorpayService.getprofile(body);
  }
  
  @Post('/freetrail')
  async freeTrail(@Request() req,@Body() body:{startDate:Date,endDate:Date,freetrail:boolean,subscriptionPlan:string}){
    const {startDate,endDate,freetrail,subscriptionPlan}=body
    console.log(req.user)
    
    const {email,picture}=req.user
    return this.razorpayService.freeTrail(email,startDate,endDate,freetrail,subscriptionPlan,picture)
  }
  @Post('order')
  async createOrder(@Body() body: { amount: number }) {
    const { amount } = body;
    return await this.razorpayService.createOrder(amount);
  }

  @Post('capture')
  async capturePayment(@Body() body: { paymentId: string; amount: number }) {
    const { paymentId, amount } = body;
    const captureResponse=await this.razorpayService.capturePayment(paymentId, amount);
    return captureResponse;

  }

  @Post('dbupdate')
  async dbUpdate(@Request() req,@Body() body:{isSubscriber:boolean,subscriptionPlan:string,startDate:Date,endDate:Date}){
     const {isSubscriber,subscriptionPlan,startDate,endDate}=body
     const {email,picture}=req.user
     return this.razorpayService.dbUpdate(isSubscriber,subscriptionPlan,startDate,endDate,email,picture)
  }
}
