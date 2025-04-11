const { fireModel } = require("../models/fire");

async function postFire(req, res) {
  const port = process.env.PORT;
  try {
    const data = {
      ...req.body,
      imageUrl: "http://localhost:" + port + "/image/" + req.fileName,
      status: 'pending' // Set initial status as pending
    };

    console.log(data);

    const fire = await fireModel.create(data);

    return res.json({ message: "updated" });
  } catch (error) {
    console.error("Error creating fire report:", error);
    res.status(500).json({ error: "Something went wrong" });
  }
}

async function getAllFire(req, res) {
  try {
    const result = await fireModel.find({});
    res.json(result);
  } catch (error) {
    console.error("Error getting fire reports:", error);
    res.status(500).json({ error: "Failed to fetch fire reports" });
  }
}

async function updateFireStatus(req, res) {
  try {
    const { id } = req.params;
    const { status } = req.body;

    console.log(`Attempting to update fire report ${id} to status: ${status}`);
    console.log('Request body:', req.body);

    // Validate input
    if (!id) {
      console.error('Missing ID parameter');
      return res.status(400).json({ error: "Missing ID parameter" });
    }

    if (!status) {
      console.error('Missing status in request body');
      return res.status(400).json({ error: "Missing status parameter" });
    }

    // Validate status
    if (!['approved', 'denied', 'pending'].includes(status)) {
      console.error(`Invalid status value: ${status}`);
      return res.status(400).json({ error: "Invalid status" });
    }

    // Update the fire report status
    const updatedFire = await fireModel.findByIdAndUpdate(
      id,
      { status },
      { new: true }
    );

    if (!updatedFire) {
      console.error(`Fire report not found with ID: ${id}`);
      return res.status(404).json({ error: "Fire report not found" });
    }

    console.log(`Successfully updated fire report ${id} to status: ${status}`);
    console.log('Updated fire report:', updatedFire);
    
    res.json(updatedFire);
  } catch (error) {
    console.error("Error updating fire status:", error);
    res.status(500).json({ error: error.message || "Failed to update fire status" });
  }
}

module.exports = { postFire, getAllFire, updateFireStatus };
