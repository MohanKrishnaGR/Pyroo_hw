const Joi = require('joi');
require('dotenv').config();

const envSchema = Joi.object({
  PORT: Joi.number().default(8000),
  MONGO_URI: Joi.string().required().description('MongoDB Connection URI'),
  JWT_SECRET: Joi.string().required().description('JWT Secret Key'),
  JWT_EXPIRES_IN: Joi.string().default('1d').description('JWT Expiration Time'),
  ADMIN_EMAIL: Joi.string().email().required(),
  ADMIN_PASSWORD: Joi.string().min(8).required(),
  WEATHER_API_KEY: Joi.string().allow('').optional(),
  NODE_ENV: Joi.string()
    .allow('development', 'production', 'test')
    .default('development'),
}).unknown();

const { error, value: envVars } = envSchema.validate(process.env);

if (error) {
  throw new Error(`Config validation error: ${error.message}`);
}

module.exports = {
  env: envVars.NODE_ENV,
  port: envVars.PORT,
  mongoUri: envVars.MONGO_URI,
  jwt: {
    secret: envVars.JWT_SECRET,
    expiresIn: envVars.JWT_EXPIRES_IN,
  },
  admin: {
    email: envVars.ADMIN_EMAIL,
    password: envVars.ADMIN_PASSWORD,
  },
  weatherApiKey: envVars.WEATHER_API_KEY,
};
