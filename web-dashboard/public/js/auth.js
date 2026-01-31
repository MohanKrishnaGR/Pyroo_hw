class Auth {
  constructor() {
    this.loginForm = document.getElementById('adminLoginForm');
    this.usernameInput = document.getElementById('username');
    this.passwordInput = document.getElementById('password');
    this.init();
  }

  init() {
    if (this.loginForm) {
      this.loginForm.addEventListener('submit', (e) => {
        e.preventDefault();
        this.login();
      });
    }
  }

  login() {
    const username = this.usernameInput.value;
    const password = this.passwordInput.value;

    if (!username || !password) {
      Notification.show('Please enter both username and password', 'error');
      return;
    }

    Notification.showLoading();

    fetch('/login', {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({
        email: username,
        password: password,
      }),
    })
      .then((response) => response.json())
      .then((data) => {
        Notification.hideLoading();
        if (data.success) {
          // Store auth data
          localStorage.setItem('token', data.token);
          localStorage.setItem('user', JSON.stringify(data.user));

          Notification.show('Login successful!', 'success');
          window.location.href = '/admin.html';
        } else {
          Notification.show(
            data.message || 'Invalid credentials. Please try again.',
            'error'
          );
        }
      })
      .catch((error) => {
        console.error('Error:', error);
        Notification.hideLoading();
        Notification.show('Error during login. Please try again.', 'error');
      });
  }

  static logout() {
    // Add logout functionality if needed
    localStorage.removeItem('adminToken');
    window.location.href = '/';
  }
}

// Export the class
window.Auth = Auth;
