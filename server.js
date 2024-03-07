const express = require("express");
const bodyParser = require("body-parser");
const path = require("path"); // Import the 'path' module
const { connectDb } = require("./utils/connectDb");
const { postFire } = require("./controllers/fire.controller");
const { fireRouter } = require("./routes/fire.route");

require("dotenv").config();

connectDb();

const app = express();
const port = process.env.PORT;

// Hard-coded array of predefined email and password pairs
const users = [
  { email: "user1@example.com", password: "password1" },
  { email: "user2@example.com", password: "password2" },
  // Add more users as needed
];

// Middleware to parse JSON in the request body
app.use(bodyParser.json());

// Serve static files from the 'public' directory
app.use(express.static(path.join(__dirname, "public")));
app.use("/image/", express.static(path.join(__dirname, "uploads")));

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

app.listen(port, () => {
  console.log(`Server is running at http://localhost:${port}`);
});
