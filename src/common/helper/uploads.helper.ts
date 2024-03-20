
import * as fs from 'fs';
import { promisify } from 'util';

export const googleclientId='226507905282-noov74vi8s3fn38k31449vjvtmn17a74.apps.googleusercontent.com'

export const checkIfFileOrDirectoryExists = (path: string): boolean => {
    return fs.existsSync(path);
  };

export const createFile = async (
    path: string,
    fileName: string,
    data: string,
  ): Promise<void> => {
    if (!checkIfFileOrDirectoryExists(path)) {
      fs.mkdirSync(path);
    }
  
    const writeFile = promisify(fs.writeFile);
  
    return await writeFile(`${path}/${fileName}`, data, 'utf8');
  };
  
export const createDir = (
    path: string
  ): void => {
    if (!checkIfFileOrDirectoryExists(path)) {
      fs.mkdirSync(path);
    }
  };