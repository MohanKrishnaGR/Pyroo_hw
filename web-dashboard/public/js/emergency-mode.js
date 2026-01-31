class EmergencyMode {
  constructor() {
    this.active = false;
    this.emergencyModeToggle = document.getElementById('emergencyModeToggle');
    this.emergencyHeader = document.getElementById('emergencyHeader');
    this.emergencyContacts = document.getElementById('emergencyContacts');
    this.exitButton = document.getElementById('emergencyExitButton');
    this.body = document.body;

    this.init();
  }

  init() {
    this.emergencyModeToggle.addEventListener('click', (e) => {
      e.preventDefault();
      this.toggle();
    });

    this.exitButton.addEventListener('click', (e) => {
      e.preventDefault();
      if (confirm('Are you sure you want to exit Emergency Mode?')) {
        this.deactivate();
      }
    });

    document.addEventListener('keydown', (e) => {
      if (e.key === 'Escape' && this.active) {
        e.preventDefault();
        if (confirm('Are you sure you want to exit Emergency Mode?')) {
          this.deactivate();
        }
      }
    });
  }

  toggle() {
    if (!this.active) {
      this.activate();
    }
  }

  activate() {
    this.active = true;
    this.body.setAttribute('data-emergency', 'true');
    this.emergencyHeader.style.display = 'block';
    this.emergencyContacts.style.display = 'block';
    this.emergencyModeToggle.style.display = 'none';
    this.exitButton.style.display = 'flex';

    // Play emergency sound
    const audio = new Audio('emergency-alert.mp3');
    audio.play();

    Notification.show('Emergency Mode Activated!', 'error');
  }

  deactivate() {
    this.active = false;
    this.body.setAttribute('data-emergency', 'false');
    this.emergencyHeader.style.display = 'none';
    this.emergencyContacts.style.display = 'none';
    this.emergencyModeToggle.style.display = 'flex';
    this.exitButton.style.display = 'none';

    Notification.show('Emergency Mode Deactivated', 'success');
  }

  static callEmergency(type) {
    const numbers = {
      fire: '101',
      police: '100',
      ambulance: '108',
      disaster: '1077',
      women: '1091',
      child: '1098',
      national: '112',
    };

    if (numbers[type]) {
      window.location.href = `tel:${numbers[type]}`;
    }
  }
}

// Export the class
window.EmergencyMode = EmergencyMode;
