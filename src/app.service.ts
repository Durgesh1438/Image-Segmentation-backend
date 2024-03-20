/* eslint-disable @typescript-eslint/no-var-requires */
/* eslint-disable @typescript-eslint/no-unused-vars */
import { Injectable,NotFoundException } from '@nestjs/common';
import {spawn} from 'child_process'
import { checkIfFileOrDirectoryExists } from './common/helper/uploads.helper';
import * as fs from 'fs'
const storageFolderName = 'storage';
@Injectable()
export class AppService {
  getHello(): string {
    return 'Backend called';
  }

  async processImage(unique_id: string, CoinDiaEntry : number): Promise<{ ppm: number; filename: string }> {
    const output_filepath = `${storageFolderName}/${unique_id}` + '/calibrated_img.jpg';
    if (checkIfFileOrDirectoryExists(output_filepath)) {
      fs.rmSync(output_filepath);
    }
    if (!checkIfFileOrDirectoryExists(storageFolderName)) {
      fs.mkdirSync(storageFolderName);
      console.log('Created storage directory');
    }
    if (!checkIfFileOrDirectoryExists(`${storageFolderName}/${unique_id}`)) {
      fs.mkdirSync(`${storageFolderName}/${unique_id}`);
      console.log('Created static folder');
    }
    try {
      // Run the Python script and pass the image path

      const pythonProcess = require('child_process').spawn('python',
       ['./src/py-scripts/calibrate.py', 
       `/${storageFolderName}/` + unique_id + '/original_img.jpg',
        CoinDiaEntry.toString(),
        '/' + output_filepath,
      ]
      );

      // Capture output from the Python script
      let output: string = '';
      pythonProcess.stdout.on('data', (data) => {
         output= data.toString();
      });
      // Handle errors from the Python script
      pythonProcess.stderr.on('data', (data) => {
        console.error('Python script error:', data.toString());
      });

      // Wait for the Python script to finish
      await new Promise<void>((resolve, reject) => {
        pythonProcess.on('close', (code) => {
          if (code === 0) {
            resolve();
          } else {
            reject(new Error(`Python script exited with code ${code}`));
          }
        });
      });
      const lines = output.split('\n');
      const ppm = parseFloat(lines[0]);
      const filename = lines[1];
      console.log(lines)
      console.log(ppm)
      console.log(filename)

      return { ppm, filename };

      // Return the processed image data
      // return processedImage
      //return output;
    } catch (error) {
      console.error('Error:', error);
      throw new Error('Error processing image');
    }
  }
  async processColorCluster(unique_id:string,minAreaValue: number, maxAreaValue: number, cluster:number): Promise<{colorclusteredimg:string,excelpath1:string,excelpath2:string}> {
    const excelfilename1=`/${storageFolderName}/`+unique_id+`/color_clustered1.xlsx`
    const excelfilename2=`/${storageFolderName}/`+unique_id+`/color_clustered2.xlsx`
    const output_filepath = `${storageFolderName}/${unique_id}` + '/color_clustered_img.jpg';
    if (checkIfFileOrDirectoryExists(output_filepath)) {
      fs.rmSync(output_filepath);
    }
    if (!checkIfFileOrDirectoryExists(storageFolderName)) {
      // createDir(`${storageFolderName}/static`);
      fs.mkdirSync(storageFolderName);
      console.log('Created storage directory');
    }
    if (!checkIfFileOrDirectoryExists(`${storageFolderName}/${unique_id}`)) {
      fs.mkdirSync(`${storageFolderName}/${unique_id}`);
      console.log('Created static folder');
    }
    
    const original_img = `/${storageFolderName}/` + unique_id + '/original_img.jpg';
    const output_path = '/' + output_filepath;
    try {
      // Your color clustering logic goes here
      // You can use the minAreaValue, maxAreaValue, and cluster parameters to perform clustering
      const pythonProcess = spawn('python',
       ['./src/py-scripts/colorcluster.py', 
       original_img ,
       minAreaValue.toString(),
       maxAreaValue.toString(),
       cluster.toString(),
       excelfilename1,
       excelfilename2,
       output_path
      ]
      );
      let output: string = '';
      pythonProcess.stdout.on('data', (data) => {
         output= data.toString();
      });
      // Handle errors from the Python script
      pythonProcess.stderr.on('data', (data) => {
        console.error('Python script error:', data.toString());
      });

      // Wait for the Python script to finish
      await new Promise<void>((resolve, reject) => {
        pythonProcess.on('close', (code) => {
          if (code === 0) {
            resolve();
          } else {
            reject(new Error(`Python script exited with code ${code}`));
          }
        });
      });
      //console.log(output)
      const lines = output.split('\n').map(line => line.replace(/\r/g, '')).filter(line => line !== '');
      const colorclusteredimg=lines[0]
      const excelpath1=lines[1]
      const excelpath2=lines[2]
      console.log(colorclusteredimg)
      console.log(excelpath1)
      console.log(excelpath2)
      return {colorclusteredimg,excelpath1,excelpath2}
      // For demonstration purposes, let's assume we're just returning the provided parameters
      //return { minAreaValue, maxAreaValue, cluster};
    } catch (error) {
      // Handle any errors that occur during the color clustering process
      console.error('Error processing color clustering:', error);
      throw new Error('Error processing color clustering');
    }
  }

  async processMorphCluster(unique_id:string,minAreaValue: string, maxAreaValue: string, cluster:string,ppmm:number,parameterSelection:string): Promise<{morphclusteredimg:string,excelpath1:string,excelpath2:string}> {
    const excelfilename1=`/${storageFolderName}/`+unique_id+`/morph_clustered1.xlsx`
    const excelfilename2=`/${storageFolderName}/`+unique_id+`/morph_clustered2.xlsx`
    const output_filepath = `${storageFolderName}/` + unique_id + '/morph_clustered_img.jpg';
    if (checkIfFileOrDirectoryExists(output_filepath)) {
      fs.rmSync(output_filepath);
    }
    if (!checkIfFileOrDirectoryExists(storageFolderName)) {
      // createDir(`${storageFolderName}/static`);
      fs.mkdirSync(storageFolderName);
      console.log('Created storage directory');
    }
    if (!checkIfFileOrDirectoryExists(`${storageFolderName}/${unique_id}`)) {
      fs.mkdirSync(`${storageFolderName}/${unique_id}`);
      console.log('Created static folder');
    }
    const original_img = `/${storageFolderName}/` + unique_id + '/original_img.jpg';
    const output_path = '/' + output_filepath;
    try {
      // Your color clustering logic goes here
      // You can use the minAreaValue, maxAreaValue, and cluster parameters to perform clustering
      console.log(parameterSelection)
      const pythonProcess = spawn('python',
       ['./src/py-scripts/morph_cluster.py', 
       original_img,
       minAreaValue,
       maxAreaValue,
       cluster,
       ppmm.toString(),
       parameterSelection,
       excelfilename1,
       excelfilename2,
       output_path
      ]
      );
      let output: string = '';
      pythonProcess.stdout.on('data', (data) => {
         output= data.toString();
      });
      // Handle errors from the Python script
      pythonProcess.stderr.on('data', (data) => {
        console.error('Python script error:', data.toString());
      });

      // Wait for the Python script to finish
      await new Promise<void>((resolve, reject) => {
        pythonProcess.on('close', (code) => {
          if (code === 0) {
            resolve();
          } else {
            reject(new Error(`Python script exited with code ${code}`));
          }
        });
      });
      //console.log(output)
      const lines = output.split('\n').map(line => line.replace(/\r/g, '')).filter(line => line !== '');
      console.log(lines)
      const morphclusteredimg=lines[0]
      const excelpath1=lines[1]
      const excelpath2=lines[2]
      console.log(morphclusteredimg)
      console.log(excelpath1)
      console.log(excelpath2)
      return {morphclusteredimg,excelpath1,excelpath2}
      // For demonstration purposes, let's assume we're just returning the provided parameters
      //return { minAreaValue, maxAreaValue, cluster};
      //return {minAreaValue,maxAreaValue,parameterSelection}
    } catch (error) {
      // Handle any errors that occur during the color clustering process
      console.error('Error processing color clustering:', error);
      throw new Error('Error processing color clustering');
    }
  }

  async processMeasureCluster(unique_id:string,minAreaValue: number, maxAreaValue: number, cluster:number,ppmm:number,parameterSelection:string): Promise<{measureclusteredimg:string,excelpath3:string}> {
    const excelfilename1=`/${storageFolderName}/`+unique_id+`/measure_clustered1.xlsx`
    const excelfilename2=`/${storageFolderName}/`+unique_id+`/measure_clustered2.xlsx`
    const excelfilename3 = `/${storageFolderName}/`+unique_id +`/grains.xlsx`;
    const output_filepath = `${storageFolderName}/`+unique_id +'/measure_seed_img.jpg';
    if (checkIfFileOrDirectoryExists(output_filepath)) {
      fs.rmSync(output_filepath);
    }
    if (!checkIfFileOrDirectoryExists(storageFolderName)) {
      fs.mkdirSync(storageFolderName);
      console.log('Created storage directory');
    }
    if (!checkIfFileOrDirectoryExists(`${storageFolderName}/${unique_id}`)) {
      fs.mkdirSync(`${storageFolderName}/${unique_id}`);
      console.log('Created static folder');
    }
    const original_img = `/${storageFolderName}/` + unique_id + '/original_img.jpg';
    const output_path = '/' + output_filepath;
    try {
      // Your color clustering logic goes here
      // You can use the minAreaValue, maxAreaValue, and cluster parameters to perform clustering
      
      
      const pythonProcess = spawn('python',
       ['./src/py-scripts/measure_seed.py', 
       original_img,
       minAreaValue.toString(),
       maxAreaValue.toString(),
       cluster.toString(),
       ppmm.toString(),
       parameterSelection,
       excelfilename1,
       excelfilename2,
       excelfilename3,
       output_path
      ]
      );
      let output: string = '';
      pythonProcess.stdout.on('data', (data) => {
         output= data.toString();
      });
      // Handle errors from the Python script
      pythonProcess.stderr.on('data', (data) => {
        console.error('Python script error:', data.toString());
      });

      // Wait for the Python script to finish
      await new Promise<void>((resolve, reject) => {
        pythonProcess.on('close', (code) => {
          if (code === 0) {
            resolve();
          } else {
            reject(new Error(`Python script exited with code ${code}`));
          }
        });
      });
      //console.log(output)
      const lines = output.split('\n').map(line => line.replace(/\r/g, '')).filter(line => line !== '');
      console.log(lines)
      const measureclusteredimg=lines[0]
      const excelpath3=lines[1]
      console.log(measureclusteredimg)
      console.log(excelpath3)
      return {measureclusteredimg,excelpath3}
      // For demonstration purposes, let's assume we're just returning the provided parameters
      //return { minAreaValue, maxAreaValue, cluster};
      //return {minAreaValue,maxAreaValue,parameterSelection}
    } catch (error) {
      // Handle any errors that occur during the color clustering process
      console.error('Error processing color clustering:', error);
      throw new Error('Error processing color clustering');
    }
  }

  async  generateReport(unique_id:string,minAreaValue:string, maxAreaValue: string, cluster:string,ppmm:string,parameterSelection:string,testId:string,date:string,organization:string,variety:string,comments:string,email:string,name:string,processname:string): Promise<{pdf_filepath:string}> {
    //const excelfilename=`/${storageFolderName}/`+`/generatedReport.xlsx`
    const output_filepath = `/${storageFolderName}/${unique_id}/final_report.pdf`;
    const config = {
      filePrefix: 'Graseed_',
      coinImageExtension: 'Coin_',
      seedImageExtension: 'Seed_',
      ppmm,
      minAreaValue,
      maxAreaValue,
      cluster,
      parameterSelection,
      testId,
      date,
      variety,
      name,
      organization,
      email,
      comments,
      imagepath:`/${storageFolderName}/${unique_id}/original_img.jpg`,
      saveFolder: `${storageFolderName}/${unique_id}/`,
      pdfFile: output_filepath,
      processname:processname

    }
    if (checkIfFileOrDirectoryExists(output_filepath)) {
      console.log('Entered the block') 
      fs.rmSync(output_filepath);
    }
    if (!checkIfFileOrDirectoryExists(storageFolderName)) {
      fs.mkdirSync(storageFolderName);
      console.log('Created storage directory');
    }
    if (!checkIfFileOrDirectoryExists(`${storageFolderName}/${unique_id}`)) {
      fs.mkdirSync(`${storageFolderName}/${unique_id}`);
      console.log('Created static folder');
    }
    try {
      // Your color clustering logic goes here
      // You can use the minAreaValue, maxAreaValue, and cluster parameters to perform clustering
      
      const pythonProcess = spawn('python',
       ['./src/py-scripts/generatepdf.py', 
        config.filePrefix || 'prefix_',
        config.coinImageExtension || 'coin_',
        config.seedImageExtension || 'seed_',
        config.ppmm || '78',
        config.minAreaValue || '100',
        config.maxAreaValue || '1000',
        config.cluster || '2',
        config.parameterSelection || 'lengths',
        config.testId || 'test',
        config.date,
        config.variety || 'variety',
        config.name || 'Name',
        config.organization || 'organization',
        config.email || 'Email@address.com',
        config.comments || 'These are the comments',
        config.imagepath,
        config.saveFolder,
        config.pdfFile,
        config.processname
      ]
      );
      let output: string = '';
      pythonProcess.stdout.on('data', (data) => {
         output= data.toString();
      });
      // Handle errors from the Python script
      pythonProcess.stderr.on('data', (data) => {
        console.error('Python script error:', data.toString());
      });

      // Wait for the Python script to finish
      await new Promise<void>((resolve, reject) => {
        pythonProcess.on('close', (code) => {
          if (code === 0) {
            resolve();
          } else {
            reject(new Error(`Python script exited with code ${code}`));
          }
        });
      });
      //console.log(output)
      const lines = output.split('\n').map(line => line.replace(/\r/g, '')).filter(line => line !== '');
      console.log(lines)
      const pdf_filepath=lines[0]
      
      console.log(pdf_filepath)
      return {pdf_filepath}
      // For demonstration purposes, let's assume we're just returning the provided parameters
      //return { minAreaValue, maxAreaValue, cluster};
      //return {minAreaValue,maxAreaValue,parameterSelection}
    } catch (error) {
      // Handle any errors that occur during the color clustering process
      console.error('Error generating pdf:', error);
      throw new Error('Error generating pdf');
    }
  }


}

