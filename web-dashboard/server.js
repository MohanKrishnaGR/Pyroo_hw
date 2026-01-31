const express = require('express');
const path = require('path');
const helmet = require('helmet');
const cors = require('cors');
const rateLimit = require('express-rate-limit');
const config = require('./config/config');

// ... (existing code)

const app = express();

// Rate limiting
const authLimiter = rateLimit({
  windowMs: 15 * 60 * 1000, // 15 minutes
  max: 100, // Limit each IP to 100 requests per window
  message: 'Too many login attempts, please try again later',
});

// Security Middleware
app.use(helmet());
app.use(cors());
app.use('/login', authLimiter);

// Body Parsing
app.use(express.json());
app.use(express.urlencoded({ extended: true }));

// Serve static files
app.use(express.static(path.join(__dirname, 'public')));
app.use('/image/', express.static(path.join(__dirname, 'uploads')));

// Routes
app.use('/', authRouter); // Mounts /login
app.use('/fire', fireRouter);
app.get('/api/config/weather', (req, res) => {
  res.json({ apiKey: config.weatherApiKey });
});

// Default Route
app.get('/', (req, res) => {
  res.sendFile(path.join(__dirname, 'public', 'index.html'));
});

// Error handling
app.use((err, req, res, next) => {
  console.error('Error:', err.stack);
  res.status(500).json({ error: err.message || 'Something went wrong!' });
});

// 404
app.use((req, res) => {
  res.status(404).json({ error: 'Route not found' });
});

app.listen(config.port, () => {
  console.log(`Server is running at http://localhost:${config.port}`);
});
