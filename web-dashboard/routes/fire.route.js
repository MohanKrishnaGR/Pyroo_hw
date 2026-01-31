const {
  postFire,
  getAllFire,
  updateFireStatus,
} = require('../controllers/fire.controller');
const { upload } = require('../middlewares/upload');
const { protect } = require('../middlewares/auth.middleware');

const router = require('express').Router();

// POST route for creating new fire reports (Public)
router.post('/', upload.single('file'), postFire);

// GET route for retrieving all fire reports (Protected)
router.get('/', protect, getAllFire);

// PATCH route for updating fire report status (Protected)
router.patch('/:id', protect, updateFireStatus);

// Export the router directly
module.exports = router;
