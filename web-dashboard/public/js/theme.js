class ThemeManager {
  constructor() {
    this.themeToggle = document.getElementById('themeToggle');
    this.body = document.body;
    this.init();
  }

  init() {
    this.themeToggle.addEventListener('click', () => this.toggle());

    // Load saved theme preference
    const savedTheme = localStorage.getItem('theme');
    if (savedTheme) {
      this.setTheme(savedTheme);
    }
  }

  toggle() {
    const currentTheme = this.body.getAttribute('data-theme');
    const newTheme = currentTheme === 'dark' ? 'light' : 'dark';
    this.setTheme(newTheme);
  }

  setTheme(theme) {
    this.body.setAttribute('data-theme', theme);
    this.themeToggle.innerHTML =
      theme === 'dark'
        ? '<i class="fas fa-sun" aria-hidden="true"></i>'
        : '<i class="fas fa-moon" aria-hidden="true"></i>';

    // Save theme preference
    localStorage.setItem('theme', theme);
  }
}

// Export the class
window.ThemeManager = ThemeManager;
