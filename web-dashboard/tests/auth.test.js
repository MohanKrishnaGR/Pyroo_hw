const request = require('supertest');
const express = require('express');
const mongoose = require('mongoose');
const authRouter = require('../routes/auth.route');
const User = require('../models/user.model');
const bodyParser = require('body-parser');

const app = express();
app.use(bodyParser.json());
app.use('/', authRouter);

describe('Auth Endpoints', () => {
  beforeEach(async () => {
    // Create a test user
    await User.create({
      email: 'test@example.com',
      password: 'password123',
    });
  });

  it('should login successfully with correct credentials', async () => {
    const res = await request(app)
      .post('/login')
      .send({
        email: 'test@example.com',
        password: 'password123',
      });

    expect(res.statusCode).toEqual(200);
    expect(res.body).toHaveProperty('token');
    expect(res.body).toHaveProperty('success', true);
  });

  it('should fail with incorrect password', async () => {
    const res = await request(app)
      .post('/login')
      .send({
        email: 'test@example.com',
        password: 'wrongpassword',
      });

    expect(res.statusCode).toEqual(401);
    expect(res.body.success).toBe(false);
  });

  it('should fail with non-existent user', async () => {
    const res = await request(app)
      .post('/login')
      .send({
        email: 'nobody@example.com',
        password: 'password123',
      });

    expect(res.statusCode).toEqual(401);
  });
});
