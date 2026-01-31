const mongoose = require("mongoose");

async function connectDb() {
  const mongoUri = process.env.MONGO_URI;

  await mongoose.connect(mongoUri);

  console.log("mongodb connected");
}

module.exports = { connectDb };
