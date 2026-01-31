# PyroGuardian Web Dashboard

The control plane for the PyroGuardian system. This Node.js application manages user authentication, real-time fire reporting, and communicates with the MongoDB database.

## 🛠 Tech Stack
*   **Runtime:** Node.js (v18)
*   **Framework:** Express.js
*   **Database:** MongoDB (via Mongoose)
*   **Frontend:** Server-side rendered HTML/JS (Vanilla)
*   **Testing:** Jest, Supertest

## 🚀 Developer Quick Start

### Prerequisites
*   Node.js v18+
*   MongoDB (Local or Atlas)

### 1. Setup Environment
```bash
cp .env.example .env
# Edit .env with your local Mongo URI and secrets
```

### 2. Install Dependencies
```bash
npm install
```

### 3. Run Locally
```bash
# Development mode (Nodemon)
npm run dev

# Production mode
npm start
```

### 4. Run Tests
```bash
npm test
```

## 📂 Project Structure
*   `controllers/`: Request logic (Auth, Fire Reports).
*   `models/`: Mongoose schemas.
*   `routes/`: API endpoint definitions.
*   `public/`: Static assets (HTML, CSS, Client-side JS).
*   `middlewares/`: Auth verification, Upload handling.

## 🔒 Security
*   **Helmet:** Sets secure HTTP headers.
*   **Rate Limiting:** Protects `/login` against brute-force attacks.
*   **JWT:** Stateless authentication.
