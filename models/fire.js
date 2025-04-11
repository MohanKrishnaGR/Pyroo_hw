const mongoose = require("mongoose");

const schema = mongoose.Schema(
  {
    imageUrl: String,
    phone: String,
    latitude: String,
    longitude: String,
    status: {
      type: String,
      enum: ['pending', 'approved', 'denied'],
      default: 'pending'
    }
  },
  { collection: "fire", timestamp: true }
);

const model = mongoose.model("fire", schema);

module.exports = { fireModel: model };
