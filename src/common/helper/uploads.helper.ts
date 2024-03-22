
import * as fs from 'fs';
import { promisify } from 'util';

export const googleclientId='846813517923-heolcnivb3oh6i5vc5f15kntk2nbpar3.apps.googleusercontent.com'

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