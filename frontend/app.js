const API_BASE_URL = window.location.origin;

const diseaseDescriptions = {
    "Apple___Apple_scab": "A fungal disease that causes dark, scabby spots on apple leaves and fruits. Severe infections may cause leaves to fall prematurely.",
    "Apple___Black_rot": "A fungal disease that can cause leaf spots, fruit rot, and branch cankers. Infected leaves may develop circular brown areas.",
    "Apple___Cedar_apple_rust": "A fungal disease that causes bright yellow-orange spots on apple leaves. It alternates between apple trees and certain juniper or cedar plants.",
    "Apple___healthy": "The apple leaf appears healthy, with no obvious symptoms of the diseases covered by this classifier.",
    "Blueberry___healthy": "The blueberry leaf appears healthy, with no obvious symptoms of the diseases covered by this classifier.",
    "Cherry_(including_sour)___Powdery_mildew": "A fungal disease that produces a white, powdery coating on cherry leaves and shoots. Infected leaves may curl or become distorted.",
    "Cherry_(including_sour)___healthy": "The cherry leaf appears healthy, with no obvious symptoms of the diseases covered by this classifier.",
    "Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot": "A fungal disease that causes elongated, rectangular gray or tan lesions on corn leaves, potentially reducing photosynthesis.",
    "Corn_(maize)___Common_rust_": "A fungal disease that produces reddish-brown, rust-like pustules on corn leaves. Severe infections can reduce plant vigor.",
    "Corn_(maize)___Northern_Leaf_Blight": "A fungal disease that causes long, cigar-shaped gray-green or tan lesions on corn leaves. Severe infections can reduce yield.",
    "Corn_(maize)___healthy": "The corn leaf appears healthy, with no obvious symptoms of the diseases covered by this classifier.",
    "Grape___Black_rot": "A fungal disease that causes brown leaf spots with dark borders and can produce shriveled, blackened grapes.",
    "Grape___Esca_(Black_Measles)": "A grapevine disease associated with wood-infecting fungi. Leaves may show interveinal discoloration, while berries can develop dark spots.",
    "Grape___Leaf_blight_(Isariopsis_Leaf_Spot)": "A fungal disease that causes reddish-brown or dark lesions on grape leaves. Severe infections may lead to leaf drying and premature drop.",
    "Grape___healthy": "The grape leaf appears healthy, with no obvious symptoms of the diseases covered by this classifier.",
    "Orange___Haunglongbing_(Citrus_greening)": "Citrus greening is a serious disease associated with the bacterium Candidatus Liberibacter. Symptoms may include blotchy yellowing, uneven leaf coloration, and misshapen fruit.",
    "Peach___Bacterial_spot": "A bacterial disease that causes small, dark, water-soaked spots on peach leaves. Spots may enlarge, develop yellow margins, and lead to leaf drop.",
    "Peach___healthy": "The peach leaf appears healthy, with no obvious symptoms of the diseases covered by this classifier.",
    "Pepper,_bell___Bacterial_spot": "A bacterial disease that causes small, dark, water-soaked spots on pepper leaves. Severe infections may cause yellowing and premature leaf drop.",
    "Pepper,_bell___healthy": "The bell pepper leaf appears healthy, with no obvious symptoms of the diseases covered by this classifier.",
    "Potato___Early_blight": "A fungal disease that causes brown leaf spots, often with concentric rings resembling a target. Older leaves are commonly affected first.",
    "Potato___Late_blight": "A destructive disease caused by Phytophthora infestans. It can produce water-soaked, dark lesions on leaves that spread rapidly in favorable conditions.",
    "Potato___healthy": "The potato leaf appears healthy, with no obvious symptoms of the diseases covered by this classifier.",
    "Raspberry___healthy": "The raspberry leaf appears healthy, with no obvious symptoms of the diseases covered by this classifier.",
    "Soybean___healthy": "The soybean leaf appears healthy, with no obvious symptoms of the diseases covered by this classifier.",
    "Squash___Powdery_mildew": "A fungal disease that produces white, powdery patches on squash leaves. Infected leaves may yellow, dry out, and lose photosynthetic capacity.",
    "Strawberry___Leaf_scorch": "A fungal disease that causes small purple or reddish spots on strawberry leaves. Spots may enlarge, develop brown centers, and make leaves appear scorched.",
    "Strawberry___healthy": "The strawberry leaf appears healthy, with no obvious symptoms of the diseases covered by this classifier.",
    "Tomato___Bacterial_spot": "A bacterial disease that causes small, dark, water-soaked spots on tomato leaves. Lesions may enlarge and develop yellowing around them.",
    "Tomato___Early_blight": "A fungal disease that causes brown leaf spots with characteristic concentric rings. Symptoms often begin on older leaves.",
    "Tomato___Late_blight": "A destructive disease caused by Phytophthora infestans. It can cause rapidly spreading, water-soaked or dark lesions on tomato leaves.",
    "Tomato___Leaf_Mold": "A fungal disease that commonly produces pale yellow patches on the upper leaf surface and olive-green or brown mold on the underside.",
    "Tomato___Septoria_leaf_spot": "A fungal disease that causes numerous small, circular leaf spots with gray or tan centers and dark borders. Severe infections can cause defoliation.",
    "Tomato___Spider_mites Two-spotted_spider_mite": "Two-spotted spider mites feed on leaf cells, causing tiny pale speckles, yellowing, and sometimes fine webbing on tomato leaves.",
    "Tomato___Target_Spot": "A fungal disease that causes brown leaf lesions with concentric rings resembling a target. Severe infections can cause leaf drop.",
    "Tomato___Tomato_Yellow_Leaf_Curl_Virus": "A viral disease transmitted by whiteflies. Symptoms can include upward leaf curling, yellowing, stunted growth, and reduced fruit production.",
    "Tomato___Tomato_mosaic_virus": "A viral disease that can cause mottled light and dark green patterns on tomato leaves, leaf distortion, and reduced plant growth.",
    "Tomato___healthy": "The tomato leaf appears healthy, with no obvious symptoms of the diseases covered by this classifier."
};

const fileInput = document.getElementById('fileInput');
const uploadArea = document.getElementById('uploadArea');
const uploadContent = document.getElementById('uploadContent');
const imagePreview = document.getElementById('imagePreview');
const previewImg = document.getElementById('previewImg');
const removeBtn = document.getElementById('removeBtn');
const predictBtn = document.getElementById('predictBtn');
const resultsCard = document.getElementById('resultsCard');
const predictedClassEl = document.getElementById('predictedClass');
const confidenceValueEl = document.getElementById('confidenceValue');
const confidenceFillEl = document.getElementById('confidenceFill');
const diseaseDescriptionEl = document.getElementById('diseaseDescription');
const predictionListEl = document.getElementById('predictionList');
const loadingOverlay = document.getElementById('loadingOverlay');
const errorToast = document.getElementById('errorToast');
const errorMessageEl = document.getElementById('errorMessage');
const toastClose = document.getElementById('toastClose');

const uploadMethodBtn = document.getElementById('uploadMethodBtn');
const cameraMethodBtn = document.getElementById('cameraMethodBtn');
const uploadSection = document.getElementById('uploadSection');
const cameraSection = document.getElementById('cameraSection');
const cameraVideo = document.getElementById('cameraVideo');
const cameraCanvas = document.getElementById('cameraCanvas');
const captureBtn = document.getElementById('captureBtn');
const switchCameraBtn = document.getElementById('switchCameraBtn');
const closeCameraBtn = document.getElementById('closeCameraBtn');
const capturedPreview = document.getElementById('capturedPreview');
const capturedImg = document.getElementById('capturedImg');
const retakeBtn = document.getElementById('retakeBtn');
const removeCapturedBtn = document.getElementById('removeCapturedBtn');

let selectedFile = null;
let currentStream = null;
let facingMode = 'environment';

function showError(message) {
    if (!errorToast || !errorMessageEl) return;
    errorMessageEl.textContent = message;
    errorToast.style.display = 'flex';
    setTimeout(() => { errorToast.style.display = 'none'; }, 5000);
}

function showLoading(show) {
    if (loadingOverlay) loadingOverlay.style.display = show ? 'flex' : 'none';
}

function stopCamera() {
    if (currentStream) {
        currentStream.getTracks().forEach(t => t.stop());
        currentStream = null;
    }
    if (cameraVideo) cameraVideo.srcObject = null;
}

// Upload area click
if (uploadArea && fileInput) {
    uploadArea.addEventListener('click', (e) => {
        if (e.target && (e.target.classList.contains('remove-btn') || e.target.closest('.remove-btn'))) return;
        fileInput.click();
    });
}

// File input
if (fileInput) {
    fileInput.addEventListener('change', (e) => {
        const file = e.target.files && e.target.files[0];
        if (file) handleFileSelect(file);
    });
}

// Drag and drop
if (uploadArea) {
    uploadArea.addEventListener('dragover', (e) => {
        e.preventDefault();
        e.stopPropagation();
        uploadArea.classList.add('dragover');
    });
    uploadArea.addEventListener('dragleave', (e) => {
        e.preventDefault();
        e.stopPropagation();
        uploadArea.classList.remove('dragover');
    });
    uploadArea.addEventListener('drop', (e) => {
        e.preventDefault();
        e.stopPropagation();
        uploadArea.classList.remove('dragover');
        const file = e.dataTransfer.files && e.dataTransfer.files[0];
        if (file && file.type && file.type.startsWith('image/')) {
            handleFileSelect(file);
        } else {
            showError('Please drop a valid image file');
        }
    });
}

function handleFileSelect(file) {
    if (!file || !file.type || !file.type.startsWith('image/')) {
        showError('Please select a valid image file');
        return;
    }
    if (file.size > 10 * 1024 * 1024) {
        showError('File size should be less than 10MB');
        return;
    }
    selectedFile = file;
    stopCamera();
    const reader = new FileReader();
    reader.onload = (e) => {
        if (previewImg) previewImg.src = e.target.result;
        if (uploadContent) uploadContent.style.display = 'none';
        if (imagePreview) imagePreview.style.display = 'block';
        if (predictBtn) predictBtn.disabled = false;
        if (resultsCard) resultsCard.style.display = 'none';
        if (diseaseDescriptionEl) { diseaseDescriptionEl.textContent = ''; diseaseDescriptionEl.style.display = 'none'; }
    };
    reader.onerror = () => showError('Failed to read image file');
    reader.readAsDataURL(file);
}

// Remove uploaded
if (removeBtn) {
    removeBtn.addEventListener('click', (e) => {
        e.stopPropagation();
        selectedFile = null;
        if (fileInput) fileInput.value = '';
        if (uploadContent) uploadContent.style.display = 'block';
        if (imagePreview) imagePreview.style.display = 'none';
        if (predictBtn) predictBtn.disabled = true;
        if (resultsCard) resultsCard.style.display = 'none';
        if (diseaseDescriptionEl) { diseaseDescriptionEl.textContent = ''; diseaseDescriptionEl.style.display = 'none'; }
    });
}

// Method buttons
if (uploadMethodBtn && cameraMethodBtn && uploadSection && cameraSection) {
    uploadMethodBtn.addEventListener('click', () => {
        uploadMethodBtn.classList.add('active');
        cameraMethodBtn.classList.remove('active');
        uploadSection.style.display = 'block';
        cameraSection.style.display = 'none';
        stopCamera();
        if (capturedPreview) capturedPreview.style.display = 'none';
        if (cameraVideo) cameraVideo.style.display = 'block';
    });
    cameraMethodBtn.addEventListener('click', async () => {
        cameraMethodBtn.classList.add('active');
        uploadMethodBtn.classList.remove('active');
        uploadSection.style.display = 'none';
        cameraSection.style.display = 'block';
        if (resultsCard) resultsCard.style.display = 'none';
        if (selectedFile) {
            selectedFile = null;
            if (uploadContent) uploadContent.style.display = 'block';
            if (imagePreview) imagePreview.style.display = 'none';
            if (predictBtn) predictBtn.disabled = true;
        }
        await startCamera();
    });
}

async function startCamera() {
    try {
        stopCamera();
        const constraints = { video: { facingMode: facingMode, width: { ideal: 1280 }, height: { ideal: 720 } } };
        currentStream = await navigator.mediaDevices.getUserMedia(constraints);
        if (cameraVideo) {
            cameraVideo.srcObject = currentStream;
            cameraVideo.style.display = 'block';
        }
        if (capturedPreview) capturedPreview.style.display = 'none';
    } catch (err) {
        console.error(err);
        if (err.name === 'NotAllowedError') showError('Camera permission denied. Allow camera access and try again.');
        else if (err.name === 'NotFoundError') showError('No camera found on this device.');
        else showError('Unable to access camera.');
        if (cameraMethodBtn && uploadMethodBtn) {
            cameraMethodBtn.classList.remove('active');
            uploadMethodBtn.classList.add('active');
        }
        if (uploadSection && cameraSection) {
            uploadSection.style.display = 'block';
            cameraSection.style.display = 'none';
        }
    }
}

if (switchCameraBtn) {
    switchCameraBtn.addEventListener('click', async () => {
        facingMode = facingMode === 'environment' ? 'user' : 'environment';
        await startCamera();
    });
}

if (closeCameraBtn) {
    closeCameraBtn.addEventListener('click', () => {
        stopCamera();
        if (cameraMethodBtn && uploadMethodBtn) {
            cameraMethodBtn.classList.remove('active');
            uploadMethodBtn.classList.add('active');
        }
        if (uploadSection && cameraSection) {
            uploadSection.style.display = 'block';
            cameraSection.style.display = 'none';
        }
        if (capturedPreview) capturedPreview.style.display = 'none';
        if (cameraVideo) cameraVideo.style.display = 'block';
    });
}

if (captureBtn) {
    captureBtn.addEventListener('click', () => {
        try {
            const video = cameraVideo;
            const canvas = cameraCanvas;
            if (!video || !canvas) return showError('Camera not ready');
            canvas.width = video.videoWidth || 640;
            canvas.height = video.videoHeight || 480;
            const ctx = canvas.getContext('2d');
            ctx.drawImage(video, 0, 0, canvas.width, canvas.height);
            canvas.toBlob((blob) => {
                if (!blob) return showError('Failed to capture image');
                selectedFile = new File([blob], 'capture.jpg', { type: 'image/jpeg' });
                if (capturedImg) capturedImg.src = canvas.toDataURL('image/jpeg');
                if (cameraVideo) cameraVideo.style.display = 'none';
                if (capturedPreview) capturedPreview.style.display = 'block';
                if (predictBtn) predictBtn.disabled = false;
                if (resultsCard) resultsCard.style.display = 'none';
                stopCamera();
            }, 'image/jpeg', 0.9);
        } catch (e) {
            showError('Failed to capture photo');
        }
    });
}

if (retakeBtn) {
    retakeBtn.addEventListener('click', async () => {
        if (capturedPreview) capturedPreview.style.display = 'none';
        await startCamera();
        if (predictBtn) predictBtn.disabled = true;
        if (resultsCard) resultsCard.style.display = 'none';
    });
}

if (removeCapturedBtn) {
    removeCapturedBtn.addEventListener('click', () => {
        selectedFile = null;
        if (capturedPreview) capturedPreview.style.display = 'none';
        if (cameraMethodBtn && cameraMethodBtn.classList.contains('active')) {
            startCamera();
        } else if (cameraVideo) {
            cameraVideo.style.display = 'block';
        }
        if (predictBtn) predictBtn.disabled = true;
        if (resultsCard) resultsCard.style.display = 'none';
        if (diseaseDescriptionEl) { diseaseDescriptionEl.textContent = ''; diseaseDescriptionEl.style.display = 'none'; }
    });
}

// Predict
if (predictBtn) {
    predictBtn.addEventListener('click', async () => {
        if (!selectedFile) { showError('Please select or capture an image first'); return; }
        showLoading(true);
        if (resultsCard) resultsCard.style.display = 'none';
        try {
            const formData = new FormData();
            formData.append('file', selectedFile);
            const res = await fetch(`${API_BASE_URL}/predict`, { method: 'POST', body: formData });
            if (!res.ok) {
                let msg = 'Prediction failed';
                try { const d = await res.json(); msg = d.detail || msg; } catch(e){}
                throw new Error(msg);
            }
            const result = await res.json();
            displayResults(result);
        } catch (e) {
            showError(e.message);
        } finally {
            showLoading(false);
        }
    });
}

function displayResults(result) {
    const predictedClass = result.predicted_class || '';
    if (predictedClassEl) predictedClassEl.textContent = predictedClass.replace(/_/g, ' ');
    const conf = typeof result.confidence === 'number' ? result.confidence * 100 : 0;
    if (confidenceValueEl) confidenceValueEl.textContent = conf.toFixed(2) + '%';
    if (confidenceFillEl) confidenceFillEl.style.width = conf.toFixed(2) + '%';
    if (diseaseDescriptionEl) {
        const desc = diseaseDescriptions[predictedClass];
        diseaseDescriptionEl.textContent = desc || 'No description available for this class.';
        diseaseDescriptionEl.style.display = 'block';
    }
    if (predictionListEl) {
        predictionListEl.innerHTML = '';
        if (Array.isArray(result.top_k)) {
            result.top_k.forEach((pred, idx) => {
                const item = document.createElement('div');
                item.className = 'prediction-item';
                item.style.animationDelay = (idx * 0.1) + 's';
                const name = document.createElement('div');
                name.className = 'prediction-item-name';
                name.textContent = (pred.class_name || '').replace(/_/g, ' ');
                const c = document.createElement('div');
                c.className = 'prediction-item-conf';
                c.textContent = (typeof pred.probability === 'number' ? pred.probability * 100 : 0).toFixed(2) + '%';
                item.appendChild(name); item.appendChild(c);
                predictionListEl.appendChild(item);
            });
        }
    }
    if (resultsCard) {
        resultsCard.style.display = 'block';
        resultsCard.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
    }
}

if (toastClose) toastClose.addEventListener('click', () => { if (errorToast) errorToast.style.display = 'none'; });

// Model info modal
const infoFloatBtn = document.getElementById('infoFloatBtn');
const modelInfoModal = document.getElementById('modelInfoModal');
const modalCloseBtn = document.getElementById('modalCloseBtn');
const classSearchInput = document.getElementById('classSearchInput');
const classItems = document.querySelectorAll('.class-item');

if (infoFloatBtn && modelInfoModal) {
    infoFloatBtn.addEventListener('click', () => {
        modelInfoModal.style.display = 'flex';
    });
}

if (modalCloseBtn && modelInfoModal) {
    modalCloseBtn.addEventListener('click', () => {
        modelInfoModal.style.display = 'none';
    });
}

if (modelInfoModal) {
    modelInfoModal.addEventListener('click', (e) => {
        if (e.target === modelInfoModal) {
            modelInfoModal.style.display = 'none';
        }
    });
}

if (classSearchInput) {
    classSearchInput.addEventListener('input', () => {
        const searchTerm = classSearchInput.value.toLowerCase();
        classItems.forEach(item => {
            const index = item.getAttribute('data-index') || '';
            const name = item.getAttribute('data-name') || '';
            if (index.includes(searchTerm) || name.toLowerCase().includes(searchTerm)) {
                item.style.display = 'flex';
            } else {
                item.style.display = 'none';
            }
        });
    });
}
