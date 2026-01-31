class Notification {
  static show(message, type) {
    const notification = document.getElementById('notification');
    notification.textContent = message;
    notification.className = `notification ${type}`;
    notification.style.display = 'block';

    setTimeout(() => {
      notification.style.display = 'none';
    }, 5000);
  }

  static showLoading() {
    const loadingOverlay = document.querySelector('.loading-overlay');
    loadingOverlay.style.display = 'flex';
  }

  static hideLoading() {
    const loadingOverlay = document.querySelector('.loading-overlay');
    loadingOverlay.style.display = 'none';
  }
}

// Export the class
window.Notification = Notification;
