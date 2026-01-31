document.addEventListener('DOMContentLoaded', () => {
  // Initialize form handler
  const formHandler = new FormHandler();
  formHandler.init();

  // Initialize emergency mode
  const emergencyMode = new EmergencyMode();

  // Initialize theme manager
  const themeManager = new ThemeManager();

  // Initialize auth
  const auth = new Auth();

  // Initialize modals
  const reportHistoryModal = document.getElementById('reportHistoryModal');
  const nearbyStationsModal = document.getElementById('nearbyStationsModal');

  if (reportHistoryModal) {
    reportHistoryModal.addEventListener('shown.bs.modal', loadReportHistory);
  }

  if (nearbyStationsModal) {
    nearbyStationsModal.addEventListener('shown.bs.modal', loadNearbyStations);
  }

  // Add global functions
  window.submitFireForm = () => formHandler.submitForm();
  window.getLocation = () => formHandler.getLocation();
  window.callEmergency = (type) => EmergencyMode.callEmergency(type);
  window.scrollToReport = () => {
    const reportSection = document.querySelector('.form-container');
    reportSection.scrollIntoView({ behavior: 'smooth' });
    document.querySelectorAll('.nav-link').forEach((link) => {
      link.classList.remove('active');
    });
    event.target.classList.add('active');
  };

  // Initialize voice input if available
  if ('webkitSpeechRecognition' in window) {
    setupVoiceInput();
  }

  // Initialize service worker
  if ('serviceWorker' in navigator) {
    navigator.serviceWorker
      .register('/sw.js')
      .then((registration) => {
        console.log('ServiceWorker registration successful');
      })
      .catch((err) => {
        console.log('ServiceWorker registration failed: ', err);
      });
  }
});

function loadReportHistory() {
  const reportHistoryList = document.getElementById('reportHistoryList');
  reportHistoryList.innerHTML = `
        <div class="list-group-item">
            <div class="d-flex justify-content-between">
                <span>No reports found</span>
                <small class="text-muted">Start reporting to see your history</small>
            </div>
        </div>
    `;
}

function loadNearbyStations() {
  if (navigator.geolocation) {
    navigator.geolocation.getCurrentPosition((position) => {
      const { latitude, longitude } = position.coords;
      const map = document.getElementById('nearbyStationsMap');
      map.innerHTML = `
                <iframe
                    width="100%"
                    height="100%"
                    frameborder="0"
                    scrolling="no"
                    marginheight="0"
                    marginwidth="0"
                    src="https://maps.google.com/maps?q=fire+station+near+me&z=13&output=embed">
                </iframe>
            `;
    });
  }
}

function setupVoiceInput() {
  const voiceInput = document.createElement('div');
  voiceInput.className = 'voice-input';
  voiceInput.innerHTML = `
        <button onclick="startVoiceInput()">
            <i class="fas fa-microphone"></i>
        </button>
    `;
  document.querySelector('.form-group').appendChild(voiceInput);
}

function startVoiceInput() {
  const recognition = new webkitSpeechRecognition();
  recognition.continuous = false;
  recognition.interimResults = false;

  recognition.onresult = function (event) {
    const transcript = event.results[0][0].transcript;
    Notification.show('Voice input received: ' + transcript, 'success');
  };

  recognition.start();
}
