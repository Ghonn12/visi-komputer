document.addEventListener('DOMContentLoaded', function() {
    // Elemen Kontrol
    const imageInput = document.getElementById('image-input');
    const angleSelect = document.getElementById('angle-select');

    // Elemen Pratinjau Gambar
    const originalPreviewImg = document.getElementById('original-preview-img');
    const originalPlaceholder = document.getElementById('original-placeholder');
    const grayscalePreviewImg = document.getElementById('grayscale-preview-img');
    const grayscalePlaceholder = document.getElementById('grayscale-placeholder');

    // Elemen Hasil/Status
    const resultsDisplay = document.getElementById('results-display');
    const mainPlaceholder = document.getElementById('placeholder');
    const loader = document.getElementById('loader');

    // Fungsi untuk mereset tampilan
    const resetDisplays = () => {
        // Reset pratinjau asli
        originalPreviewImg.classList.add('d-none');
        originalPreviewImg.src = '';
        originalPlaceholder.classList.remove('d-none');
        
        // Reset pratinjau grayscale
        grayscalePreviewImg.classList.add('d-none');
        grayscalePreviewImg.src = '';
        grayscalePlaceholder.classList.remove('d-none');
        grayscalePlaceholder.innerText = 'Menunggu proses...';

        // Reset hasil
        resultsDisplay.classList.add('d-none');
        mainPlaceholder.classList.remove('d-none');
        loader.classList.add('d-none');
    };

    // Fungsi untuk menampilkan pratinjau gambar asli (SEGERA setelah diunggah)
    const showOriginalPreview = (file) => {
        if (file) {
            const reader = new FileReader();
            reader.onload = (e) => {
                originalPreviewImg.src = e.target.result;
                originalPreviewImg.classList.remove('d-none');
                originalPlaceholder.classList.add('d-none');
            };
            reader.readAsDataURL(file);
        }
    };

    // Fungsi utama untuk memproses gambar
    const processImage = async () => {
        const file = imageInput.files[0];
        const angle = angleSelect.value;

        if (!file) {
            resetDisplays();
            angleSelect.disabled = true;
            return;
        }

        // Tampilkan loader dan sembunyikan placeholder/hasil lama
        mainPlaceholder.classList.add('d-none');
        resultsDisplay.classList.add('d-none');
        loader.classList.remove('d-none');

        // Tampilkan placeholder di pratinjau grayscale
        grayscalePreviewImg.classList.add('d-none');
        grayscalePlaceholder.classList.remove('d-none');
        grayscalePlaceholder.innerText = 'Memproses...';


        // Siapkan data form untuk dikirim ke backend
        const formData = new FormData();
        formData.append('image', file);
        formData.append('angle', angle);

        try {
            const response = await fetch('/process', {
                method: 'POST',
                body: formData,
            });
            
            const data = await response.json();
            loader.classList.add('d-none'); // Sembunyikan loader

            if (data.status === 'success') {
                displayResults(data);
                
                // Tampilkan gambar grayscale (BARU)
                grayscalePreviewImg.src = data.grayscale_image_url;
                grayscalePreviewImg.classList.remove('d-none');
                grayscalePlaceholder.classList.add('d-none');

            } else {
                displayError(data.message);
                grayscalePlaceholder.innerText = 'Gagal proses';
            }

        } catch (error) {
            loader.classList.add('d-none');
            displayError('Gagal terhubung ke server. Silakan coba lagi.');
            grayscalePlaceholder.innerText = 'Error';
            console.error('Error:', error);
        }
    };

    // Fungsi untuk menampilkan hasil (angka)
    const displayResults = (features) => {
        resultsDisplay.innerHTML = `
            <div class="row g-3">
                <div class="col-md-6">
                    <div class="card bg-light">
                        <div class="card-body">
                            <h6 class="card-subtitle mb-2 text-muted">Contrast</h6>
                            <p class="card-text fs-4 fw-bold">${features.contrast}</p>
                        </div>
                    </div>
                </div>
                <div class="col-md-6">
                    <div class="card bg-light">
                        <div class="card-body">
                            <h6 class="card-subtitle mb-2 text-muted">Energy</h6>
                            <p class="card-text fs-4 fw-bold">${features.energy}</p>
                        </div>
                    </div>
                </div>
                <div class="col-md-6">
                    <div class="card bg-light">
                        <div class="card-body">
                            <h6 class="card-subtitle mb-2 text-muted">Homogeneity</h6>
                            <p class="card-text fs-4 fw-bold">${features.homogeneity}</p>
                        </div>
                    </div>
                </div>
                <div class="col-md-6">
                    <div class="card bg-light">
                        <div class="card-body">
                            <h6 class="card-subtitle mb-2 text-muted">ASM (Angular Second Moment)</h6>
                            <p class="card-text fs-4 fw-bold">${features.asm}</p>
                        </div>
                    </div>
                </div>
            </div>
        `;
        resultsDisplay.classList.remove('d-none');
    };

    // Fungsi untuk menampilkan pesan error
    const displayError = (message) => {
        resultsDisplay.innerHTML = `<div class="alert alert-danger">${message}</div>`;
        resultsDisplay.classList.remove('d-none');
    };

    // Event listener untuk input gambar
    imageInput.addEventListener('change', () => {
        if (imageInput.files.length > 0) {
            angleSelect.disabled = false; // Aktifkan dropdown sudut
            showOriginalPreview(imageInput.files[0]); // Tampilkan pratinjau asli SEGERA
            processImage(); // Langsung proses untuk dapatkan grayscale + data
        } else {
            angleSelect.disabled = true;
            resetDisplays();
        }
    });

    // Event listener untuk dropdown sudut
    angleSelect.addEventListener('change', processImage);
});