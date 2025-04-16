class FormHandler {
    constructor() {
        this.form = document.getElementById('fire-form');
        this.fileInput = document.getElementById('file');
        this.phoneInput = document.getElementById('phone');
        this.imagePreview = document.getElementById('imagePreview');
        this.mapPreview = document.getElementById('mapPreview');
        this.longitude = 0;
        this.latitude = 0;
    }

    init() {
        this.setupFileInput();
        this.setupPhoneInput();
        this.setupImagePreview();
    }

    setupFileInput() {
        this.fileInput.addEventListener('change', () => {
            const file = this.fileInput.files[0];
            if (file) {
                if (!file.type.match('image/jpeg')) {
                    this.showError('fileError', 'Please upload a JPEG/JPG file');
                } else {
                    this.hideError('fileError');
                }
            }
        });
    }

    setupPhoneInput() {
        this.phoneInput.addEventListener('input', () => {
            if (!this.phoneInput.checkValidity()) {
                this.showError('phoneError', 'Please enter a valid 10-digit phone number');
            } else {
                this.hideError('phoneError');
            }
        });
    }

    setupImagePreview() {
        this.fileInput.addEventListener('change', function() {
            const file = this.files[0];
            if (file) {
                const reader = new FileReader();
                reader.onload = function(e) {
                    this.imagePreview.src = e.target.result;
                    this.imagePreview.style.display = 'block';
                }.bind(this);
                reader.readAsDataURL(file);
            }
        }.bind(this));
    }

    showError(elementId, message) {
        const errorElement = document.getElementById(elementId);
        errorElement.textContent = message;
        errorElement.style.display = 'block';
        this[elementId.replace('Error', '')].classList.add('is-invalid');
    }

    hideError(elementId) {
        const errorElement = document.getElementById(elementId);
        errorElement.style.display = 'none';
        this[elementId.replace('Error', '')].classList.remove('is-invalid');
    }

    getLocation() {
        if (navigator.geolocation) {
            this.mapPreview.style.display = 'block';
            
            navigator.geolocation.getCurrentPosition(
                (position) => {
                    this.latitude = position.coords.latitude;
                    this.longitude = position.coords.longitude;
                    
                    this.mapPreview.innerHTML = `
                        <iframe
                            width="100%"
                            height="100%"
                            frameborder="0"
                            scrolling="no"
                            marginheight="0"
                            marginwidth="0"
                            src="https://maps.google.com/maps?q=${this.latitude},${this.longitude}&z=15&output=embed">
                        </iframe>
                    `;
                    
                    Notification.show('Location captured successfully!', 'success');
                },
                (error) => {
                    Notification.show('Error getting location: ' + error.message, 'error');
                }
            );
        } else {
            Notification.show('Geolocation is not supported by this browser.', 'error');
        }
    }

    submitForm() {
        const formData = new FormData(this.form);
        formData.append("latitude", this.latitude);
        formData.append("longitude", this.longitude);

        const progressBar = document.querySelector('.progress-bar');
        progressBar.style.display = 'block';

        fetch('/fire', {
            method: 'POST',
            body: formData,
        })
        .then(response => {
            if (!response.ok) {
                throw new Error('Network response was not ok');
            }
            return response.json();
        })
        .then(data => {
            if (data.message === "updated") {
                Notification.show('Report submitted successfully!', 'success');
                this.form.reset();
                this.imagePreview.style.display = 'none';
                this.mapPreview.style.display = 'none';
            } else {
                throw new Error(data.error || 'Unknown error occurred');
            }
        })
        .catch(error => {
            console.error('Error:', error);
            Notification.show('Error submitting report: ' + error.message, 'error');
        })
        .finally(() => {
            progressBar.style.display = 'none';
            document.getElementById('uploadProgress').style.width = '0%';
        });
    }
}

// Export the class
window.FormHandler = FormHandler; 