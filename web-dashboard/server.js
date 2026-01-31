const express = require("express");
const bodyParser = require("body-parser");
const path = require("path"); // Import the 'path' module
const { connectDb } = require("./utils/connectDb");
const fireRouter = require("./routes/fire.route");

require("dotenv").config();

connectDb();

const app = express();
const port = process.env.PORT || 8000;

// Load users from environment variables or a secure source
const ADMIN_EMAIL = process.env.ADMIN_EMAIL || "admin@example.com";
const ADMIN_PASSWORD = process.env.ADMIN_PASSWORD || "admin123";

const users = [
  { email: ADMIN_EMAIL, password: ADMIN_PASSWORD }
];

// Endpoint to provide the weather API key to the frontend
app.get("/api/config/weather", (req, res) => {
  res.json({ apiKey: process.env.WEATHER_API_KEY });
});

// Middleware to parse JSON in the request body
app.use(bodyParser.json());
app.use(express.json());

// Serve static files from the 'public' directory
app.use(express.static(path.join(__dirname, "public")));
app.use("/image/", express.static(path.join(__dirname, "uploads")));

// Use the fire router
app.use("/fire", fireRouter);

// Serve your HTML file as the default route
app.get("/", (req, res) => {
  const indexPath = path.join(__dirname, "public", "index.html");
  res.sendFile(indexPath);
});

// Login endpoint
app.post("/login", (req, res) => {
  const { email, password } = req.body;

  // Check if the provided email and password match any user
  const user = users.find((u) => u.email === email && u.password === password);

  if (user) {
    res.status(200).json({ success: true, message: "Login successful!" });
  } else {
    res.status(401).json({ success: false, message: "Invalid credentials" });
  }
});

// Error handling middleware
app.use((err, req, res, next) => {
  console.error('Error:', err.stack);
  res.status(500).json({ error: err.message || 'Something went wrong!' });
});

// 404 handler - must be after all other routes
app.use((req, res) => {
  console.log('404 Not Found:', req.method, req.url);
  res.status(404).json({ error: 'Route not found' });
});

app.listen(port, () => {
  console.log(`Server is running at http://localhost:${port}`);
});
