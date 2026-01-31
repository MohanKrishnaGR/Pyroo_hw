const mongoose = require('mongoose');
const config = require('../config/config');

async function connectDb() {
  try {
    await mongoose.connect(config.mongoUri);
    console.log('MongoDB connected');
  } catch (error) {
    console.error('MongoDB connection error:', error);
    process.exit(1);
  }
}

module.exports = { connectDb };
