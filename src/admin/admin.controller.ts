import { Body, Controller, Get, Post, Request } from '@nestjs/common';
import { AdminService } from './admin.service';

@Controller('admin')
export class AdminController {
    constructor(private readonly adminService:AdminService){}

    @Get('/getUsers')
    async getUsers():Promise<any>{
       return this.adminService.getAllUsers()
    }

    @Post('/searchUser')
    async SearchUser(@Request() req,@Body() body:{email:string}):Promise<any>{
         const {email}=body;
         return this.adminService.searchUser(email)
    }

    @Post('/updateUser')
    async updateUser(@Request()req,@Body() body:{email:string,pack:string}):Promise<any>{
        const {email,pack}=body;
        return this.adminService.updateUser(email,pack);
    }
}
