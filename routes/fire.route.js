const { postFire, getAllFire } = require("../controllers/fire.controller");
const { upload } = require("../middlewares/upload");

const router = require("express").Router();

router.post("/", upload.single("file"), postFire);
router.get("/", getAllFire);

module.exports = { fireRouter: router };
