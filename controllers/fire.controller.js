const { fireModel } = require("../models/fire");

async function postFire(req, res) {
  const port = process.env.PORT;
  try {
    const data = {
      ...req.body,
      imageUrl: "http://localhost:" + port + "/image/" + req.fileName,
    };

    console.log(data);

    const fire = await fireModel.create(data);

    return res.json({ message: "updated" });
  } catch (error) {
    res.json({ error: "Something went wrong" });
  }
}

async function getAllFire(req, res) {
  const result = await fireModel.find({});

  res.json(result);
}

module.exports = { postFire, getAllFire };
