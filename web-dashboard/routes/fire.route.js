const { postFire, getAllFire, updateFireStatus } = require("../controllers/fire.controller");
const { upload } = require("../middlewares/upload");

const router = require("express").Router();

// POST route for creating new fire reports
router.post("/", upload.single("file"), postFire);

// GET route for retrieving all fire reports
router.get("/", getAllFire);

// PATCH route for updating fire report status
router.patch("/:id", updateFireStatus);

// Export the router directly
module.exports = router;
