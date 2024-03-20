
import { AppService } from './app.service';
import { Controller, Post,  Get,UploadedFile, UseInterceptors, Body , Request} from '@nestjs/common';
import { AnyFilesInterceptor, FileInterceptor } from '@nestjs/platform-express';
import { diskStorage } from 'multer';
import { Public } from './auth/auth.controller';
import * as fs from 'fs';
import { checkIfFileOrDirectoryExists } from './common/helper/uploads.helper';


@Controller()
export class AppController {
  constructor(private readonly appService: AppService) {}
  
  @Public()
  @Get("/storage")
  getHello(): string {
    return this.appService.getHello();
  }

  @Post("/upload") // API path
  @UseInterceptors(
    FileInterceptor(
      "image", // name of the field being passed
      {
        storage: diskStorage({
          filename: (req: any, file, callback) => {
            console.log(req.user.username);
            console.log(file);
            callback(null, "original_img.jpg");
          },
          destination: (req, file, callback) => {
            const folder = `storage/${req.user.username}`;

            if (!checkIfFileOrDirectoryExists("storage")) {
              // createDir('storage/static');
              fs.mkdirSync("storage");
              console.log("Created storage directory");
            }
            if (!checkIfFileOrDirectoryExists(folder)) {
              fs.mkdirSync(folder);
              console.log("Created " + folder + " folder");
            }
            callback(null, folder);
          },
        }),
      },
    ),
  )
  async upload(@UploadedFile() image) {
    //console.log(image)
    return image;
  }
  
  @Post("/calibrate")
  @UseInterceptors(AnyFilesInterceptor())
  async calibrate(@Body() body,@Request() req): Promise<{ ppm: number; filename: string }> {
    //console.log(body.CoinDiaEntry)
    try {
      const  { ppm, filename } = await this.appService.processImage(req.user.username,parseFloat(body.CoinDiaEntry));
      return {ppm,filename};
      
    } catch (error) {
      console.error('Error:', error);
      return null;
    }
    
  }

  @Post("/colorcluster")
  @UseInterceptors(AnyFilesInterceptor())
  async colorcluster(
    @Body() body,
    @Request() req,
  ): Promise<{colorclusteredimg:string,excelpath1:string,excelpath2:string}> {
    try {
      console.log(req.user)

      // Call the service module to process the data
       const {colorclusteredimg,excelpath1,excelpath2} = await this.appService.processColorCluster(
        req.user.username,
        parseFloat(body.minAreaValue), 
        parseFloat(body.maxAreaValue),
        parseFloat(body.cluster));
          // Return the processed data
       return {colorclusteredimg,excelpath1,excelpath2};
    } catch (error) {
      console.error('Error processing color clustering:', error);
    }
  }

  @Post("/morphcluster")
  @UseInterceptors(AnyFilesInterceptor())
  async morphcluster(
    @Body() body,
    @Request() req,
  ): Promise<{morphclusteredimg:string,excelpath1:string,excelpath2:string}> {
    try {
      console.log([body.parameterSelection])
     const {morphclusteredimg,excelpath1,excelpath2}=await this.appService.processMorphCluster(
      req.user.username,
      body.minAreaValue,
      body.maxAreaValue,
      body.cluster,
      parseInt(body.ppmm),
      ([body.parameterSelection] || []).join(','),
     )
      return {morphclusteredimg,excelpath1,excelpath2}
      
    } catch (error) {
      console.error('Error processing color clustering:', error);
    }
  }
  
  @Post("/measurecluster")
  @UseInterceptors(AnyFilesInterceptor())
  async measurecluster(
    @Body() body,
    @Request() req,
  ): Promise<{measureclusteredimg:string,excelpath3:string}> {
    try {
      
      console.log([body.parameterSelection])
     const {measureclusteredimg,excelpath3}=await this.appService.processMeasureCluster(
      req.user.username,
      parseInt(body.minAreaValue),
      parseInt(body.maxAreaValue),
      parseInt(body.cluster),
      parseInt(body.ppmm),
      ([body.parameterSelection] || []).join(','),
     )
      return {measureclusteredimg,excelpath3}
      
    } catch (error) {
      console.error('Error processing color clustering:', error);
    }
  }

  @Post("/generateReport")
  @UseInterceptors(AnyFilesInterceptor())
  async generateReport(
    @Body() body,
    @Request() req,
  ): Promise<{pdf_filepath:string}> {
    try {
      
     const {pdf_filepath}=await this.appService.generateReport(
      req.user.username,
      body.minAreaValue,
      body.maxAreaValue,
      body.cluster,
      body.ppmm,
      ([body.parameterSelection] || []).join(','),
      body.testId,
      body.email,
      body.comments,
      body.variety,
      body.name,
      body.organization,
      body.date,
      body.processname
     )
      return {pdf_filepath}
      
    } catch (error) {
      console.error('Error processing color clustering:', error);
    }
  }


  
}
