
import { NestFactory } from '@nestjs/core';
import { AppModule } from './app.module';
import * as express from 'express';
async function bootstrap() {
  const app = await NestFactory.create(AppModule);
  app.enableCors({
    origin: 'http://localhost:3000', // Allow requests from this origin
    methods: 'GET,POST', // Allow only specified HTTP methods
    allowedHeaders: 'Content-Type,Authorization', // Allow only specified headers
  });
  app.use(express.json())
  await app.listen(3001);
}
bootstrap();
