// ====================================
// Conversational Data Intelligence Platform
// Frontend JavaScript
// ====================================

// DOM Elements
const csvFileInput = document.getElementById('csvFile');
const uploadBtn = document.getElementById('uploadBtn');
const previewBtn = document.getElementById('previewBtn');
const fileNameDisplay = document.getElementById('fileName');
const dataPassportContent = document.getElementById('dataPassportContent');
const questionInput = document.getElementById('questionInput');
const askBtn = document.getElementById('askBtn');

// Modal elements
const previewModal = document.getElementById('previewModal');
const modalClose = document.getElementById('modalClose');
const previewTable = document.getElementById('previewTable');
const previewTableHead = document.getElementById('previewTableHead');
const previewTableBody = document.getElementById('previewTableBody');

// Global state
let currentFile = null;

// ====================================
// File Upload Functionality
// ====================================

// Show selected file name
csvFileInput.addEventListener('change', (e) => {
    const file = e.target.files[0];
    if (file) {
        currentFile = file;
        fileNameDisplay.textContent = file.name;
        fileNameDisplay.style.fontStyle = 'normal';
        fileNameDisplay.style.color = 'var(--text-primary)';
    } else {
        currentFile = null;
        fileNameDisplay.textContent = 'No file chosen';
        fileNameDisplay.style.fontStyle = 'italic';
        fileNameDisplay.style.color = 'var(--text-placeholder)';
    }
});

// Upload button click handler
uploadBtn.addEventListener('click', async () => {
    if (!currentFile) {
        showError('Please select a CSV file first');
        return;
    }

    await uploadCSV(currentFile);
});

/**
 * Upload CSV file to Flask backend
 * @param {File} file - The CSV file to upload
 */
async function uploadCSV(file) {
    // Show loading state
    uploadBtn.disabled = true;
    uploadBtn.textContent = 'Uploading...';
    
    // Clear previous data passport
    dataPassportContent.innerHTML = '<p class="placeholder-text">Processing file...</p>';

    try {
        // Create FormData and append file
        const formData = new FormData();
        formData.append('file', file);

        // Send POST request to backend
        const response = await fetch('/upload', {
            method: 'POST',
            body: formData
        });

        const data = await response.json();

        if (response.ok && data.status === 'success') {
            // Display data passport
            displayDataPassport(data.passport);
            showSuccess(data.message);
        } else {
            // Handle error response
            showError(data.message || 'Upload failed');
            dataPassportContent.innerHTML = '<p class="placeholder-text">Upload failed. Please try again.</p>';
            previewBtn.disabled = true;
        }

    } catch (error) {
        console.error('Upload error:', error);
        showError('Network error. Please check your connection.');
        dataPassportContent.innerHTML = '<p class="placeholder-text">Upload a CSV file to see data information</p>';
        previewBtn.disabled = true;
    } finally {
        // Reset button state
        uploadBtn.disabled = false;
        uploadBtn.textContent = 'Upload';
    }
}

/**
 * Display data passport in the sidebar
 * @param {Object} passport - Data passport object from backend
 */
function displayDataPassport(passport) {
    // Create HTML for data passport
    let html = `
        <div class="passport-summary">
            <div class="passport-item">
                <span class="passport-label">Total Rows:</span>
                <span class="passport-value">${passport.total_rows.toLocaleString()}</span>
            </div>
            <div class="passport-item">
                <span class="passport-label">Total Columns:</span>
                <span class="passport-value">${passport.total_columns}</span>
            </div>
        </div>
        
        <div class="passport-columns">
            <h4 style="font-size: 0.875rem; margin: 1rem 0 0.5rem 0; color: var(--text-secondary);">Columns</h4>
            <div class="columns-table">
    `;

    // Add each column information
    passport.columns.forEach((col, index) => {
        const sampleValues = col.sample_values.length > 0 
            ? col.sample_values.map(v => `"${v}"`).join(', ')
            : 'N/A';
        
        html += `
            <div class="column-row">
                <div class="column-header">
                    <strong>${col.name}</strong>
                    <span class="column-type">${col.dtype}</span>
                </div>
                <div class="column-details">
                    <span class="detail-item">Unique: ${col.unique_values}</span>
                </div>
                <div class="column-samples">
                    <span class="samples-label">Samples:</span>
                    <span class="samples-text">${sampleValues}</span>
                </div>
            </div>
        `;
    });

    html += `
            </div>
        </div>
    `;

    dataPassportContent.innerHTML = html;
    
    // Enable preview button after successful upload
    previewBtn.disabled = false;
}

// ====================================
// CSV Preview Functionality
// ====================================

// Preview button click handler
previewBtn.addEventListener('click', async () => {
    await fetchPreviewData();
});

// Modal close button
modalClose.addEventListener('click', () => {
    closeModal();
});

// Close modal when clicking outside
previewModal.addEventListener('click', (e) => {
    if (e.target === previewModal) {
        closeModal();
    }
});

// Close modal with Escape key
document.addEventListener('keydown', (e) => {
    if (e.key === 'Escape' && previewModal.classList.contains('show')) {
        closeModal();
    }
});

/**
 * Fetch preview data from backend
 */
async function fetchPreviewData() {
    try {
        previewBtn.disabled = true;
        previewBtn.textContent = 'Loading...';

        const response = await fetch('/preview');
        const data = await response.json();

        if (response.ok && data.status === 'success') {
            displayPreviewTable(data.preview);
            openModal();
        } else {
            showError(data.message || 'Failed to load preview');
        }

    } catch (error) {
        console.error('Preview error:', error);
        showError('Network error. Please try again.');
    } finally {
        previewBtn.disabled = false;
        previewBtn.textContent = 'Preview Data';
    }
}

/**
 * Display preview data in modal table
 * @param {Object} preview - Preview data with columns and rows
 */
function displayPreviewTable(preview) {
    // Clear existing table
    previewTableHead.innerHTML = '';
    previewTableBody.innerHTML = '';

    // Create header row
    const headerRow = document.createElement('tr');
    preview.columns.forEach(col => {
        const th = document.createElement('th');
        th.textContent = col;
        headerRow.appendChild(th);
    });
    previewTableHead.appendChild(headerRow);

    // Create data rows
    preview.data.forEach(row => {
        const tr = document.createElement('tr');
        row.forEach(cell => {
            const td = document.createElement('td');
            td.textContent = cell !== null && cell !== undefined ? cell : '';
            tr.appendChild(td);
        });
        previewTableBody.appendChild(tr);
    });
}

/**
 * Open preview modal
 */
function openModal() {
    previewModal.classList.add('show');
    document.body.style.overflow = 'hidden';
}

/**
 * Close preview modal
 */
function closeModal() {
    previewModal.classList.remove('show');
    document.body.style.overflow = '';
}

// ====================================
// Question Answering (Placeholder)
// ====================================

askBtn.addEventListener('click', async () => {
    const question = questionInput.value.trim();
    
    if (!question) {
        showError('Please enter a question');
        return;
    }

    await askQuestion(question);
});

// Allow Enter key to submit question
questionInput.addEventListener('keypress', (e) => {
    if (e.key === 'Enter') {
        askBtn.click();
    }
});

/**
 * Send question to backend and display answer with generated code
 * @param {string} question - User's question
 */
async function askQuestion(question) {
    // Show loading state
    askBtn.disabled = true;
    askBtn.textContent = 'Processing...';
    
    // Clear previous results and show loading
    const answerContent = document.getElementById('answerContent');
    const codeContent = document.getElementById('generatedCode');
    const explanationContent = document.getElementById('explanationContent');
    
    answerContent.innerHTML = '<p class="placeholder-text">Processing your question...</p>';
    codeContent.textContent = '// Generating code...';
    explanationContent.innerHTML = '<p class="placeholder-text">Generating explanation...</p>';

    try {
        const response = await fetch('/ask', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({ question })
        });

        const data = await response.json();

        if (response.ok && data.status === 'success') {
            // Display answer
            answerContent.innerHTML = `<p>${escapeHtml(data.answer)}</p>`;
            
            // Display explanation
            if (data.explanation) {
                explanationContent.innerHTML = `<p>${escapeHtml(data.explanation)}</p>`;
            } else {
                explanationContent.innerHTML = '<p class="placeholder-text">No explanation available</p>';
            }
            
            // Display generated code
            if (data.code) {
                codeContent.textContent = data.code;
            }
            
            // Display transparency data
            if (data.transparency) {
                displayTransparency(data.transparency);
            }
            
            // Display chart if available
            if (data.chart) {
                renderChart(data.chart);
            } else {
                // Clear previous chart if no chart data
                clearChart();
            }
            
            showSuccess('Question answered successfully!');
            
            // Clear the question input
            questionInput.value = '';
        } else if (response.ok && data.status === 'info') {
            // Handle informational messages (e.g., predictive questions not supported)
            answerContent.innerHTML = `<div style="background: #fff3cd; border-left: 4px solid #ffc107; padding: 15px; border-radius: 4px;">
                <p style="margin: 0; color: #856404;"><strong>⚠️ Notice:</strong> ${escapeHtml(data.message)}</p>
            </div>`;
            
            // Show the code that was generated (for transparency)
            if (data.code) {
                codeContent.textContent = data.code;
            } else {
                codeContent.textContent = '// No code generated';
            }
            
            // Display transparency data (even for errors)
            if (data.transparency) {
                displayTransparency(data.transparency);
            }
            
            explanationContent.innerHTML = '<p class="placeholder-text">Please rephrase your question to analyze existing data.</p>';
            clearChart();
            
            // Don't clear the input so user can easily edit the question
        } else {
            answerContent.innerHTML = '<p class="placeholder-text">Failed to get answer. Please try again.</p>';
            codeContent.textContent = '// No code generated';
            explanationContent.innerHTML = '<p class="placeholder-text">No explanation available</p>';
            clearChart();
            hideTransparency();
            showError(data.message || 'Failed to process question');
        }

    } catch (error) {
        console.error('Question error:', error);
        answerContent.innerHTML = '<p class="placeholder-text">Error occurred. Please try again.</p>';
        codeContent.textContent = '// Error occurred';
        explanationContent.innerHTML = '<p class="placeholder-text">Error occurred</p>';
        clearChart();
        hideTransparency();
        showError('Network error. Please try again.');
    } finally {
        askBtn.disabled = false;
        askBtn.textContent = 'Ask';
    }
}

// ====================================
// Transparency Display
// ====================================

/**
 * Display transparency data in the confidence & transparency section
 * @param {object} transparency - Transparency metadata object
 */
function displayTransparency(transparency) {
    const transparencyCard = document.getElementById('transparencyCard');
    
    // Update all transparency values
    document.getElementById('dataSource').textContent = transparency.data_source;
    document.getElementById('totalRows').textContent = transparency.total_rows.toLocaleString();
    
    // Columns used - display count and names
    const columnsUsed = transparency.columns_used;
    const columnsText = columnsUsed.length > 0 
        ? `${columnsUsed.length} (${columnsUsed.join(', ')})` 
        : 'None';
    document.getElementById('columnsUsed').textContent = columnsText;
    
    document.getElementById('rowsProcessed').textContent = transparency.rows_processed.toLocaleString();
    document.getElementById('analysisType').textContent = transparency.analysis_type;
    document.getElementById('complexity').textContent = transparency.complexity;
    
    // External data with checkmark
    const externalDataText = transparency.external_data_used ? 'Yes ✗' : 'No ✓';
    document.getElementById('externalData').textContent = externalDataText;
    
    // Confidence score with color coding
    const confidenceElem = document.getElementById('confidenceScore');
    const confidenceScore = transparency.confidence_score;
    confidenceElem.textContent = `${confidenceScore}%`;
    
    // Remove previous confidence classes
    confidenceElem.classList.remove('high-confidence', 'medium-confidence', 'low-confidence');
    
    // Add appropriate confidence class
    if (confidenceScore >= 85) {
        confidenceElem.classList.add('high-confidence');
    } else if (confidenceScore >= 60) {
        confidenceElem.classList.add('medium-confidence');
    } else {
        confidenceElem.classList.add('low-confidence');
    }
    
    // Execution time
    document.getElementById('executionTime').textContent = `${transparency.execution_time_ms} ms`;
    
    // Show the transparency card
    transparencyCard.style.display = 'block';
}

/**
 * Hide the transparency section
 */
function hideTransparency() {
    const transparencyCard = document.getElementById('transparencyCard');
    transparencyCard.style.display = 'none';
}

// ====================================
// Chart Rendering
// ====================================

let currentChart = null;
let currentChartData = null; // Store chart data for type switching

/**
 * Render a chart using Chart.js
 * @param {Object} chartData - Chart data from backend
 */
function renderChart(chartData) {
    console.log('Rendering chart with data:', chartData);
    
    // Store chart data globally for type switching
    currentChartData = chartData;
    
    const chartCard = document.getElementById('chartCard');
    const chartSelector = document.getElementById('chartTypeSelector');
    
    // Update dropdown options based on available chart types
    updateChartTypeOptions(chartData.available_types || ['bar']);
    
    // Set default chart type
    const defaultType = chartData.default_type || 'bar';
    chartSelector.value = defaultType;
    
    // Render the chart with default type
    renderChartWithType(defaultType);
    
    // Show the chart card
    chartCard.style.display = 'block';
    console.log('Chart rendered and card displayed');
}

/**
 * Update chart type dropdown options
 * @param {Array} availableTypes - Available chart types
 */
function updateChartTypeOptions(availableTypes) {
    const chartSelector = document.getElementById('chartTypeSelector');
    
    // Clear existing options
    chartSelector.innerHTML = '';
    
    // Chart type labels
    const chartLabels = {
        'bar': 'Bar Chart',
        'line': 'Line Chart',
        'pie': 'Pie Chart',
        'doughnut': 'Doughnut Chart',
        'horizontalBar': 'Horizontal Bar',
        'area': 'Area Chart',
        'polarArea': 'Polar Area',
        'radar': 'Radar Chart'
    };
    
    // Add available options
    availableTypes.forEach(type => {
        const option = document.createElement('option');
        option.value = type;
        option.textContent = chartLabels[type] || type;
        chartSelector.appendChild(option);
    });
}

/**
 * Render chart with specific type
 * @param {string} chartType - Chart type to render
 */
function renderChartWithType(chartType) {
    if (!currentChartData) return;
    
    const canvas = document.getElementById('dataChart');
    const ctx = canvas.getContext('2d');
    
    // Destroy previous chart if exists
    if (currentChart) {
        currentChart.destroy();
    }
    
    // Configure chart based on type
    const config = getChartConfig(chartType, currentChartData);
    
    // Create new chart
    currentChart = new Chart(ctx, config);
}

/**
 * Get chart configuration based on type
 * @param {string} chartType - Chart type
 * @param {Object} data - Chart data
 * @returns {Object} Chart.js configuration
 */
function getChartConfig(chartType, data) {
    const labels = data.labels;
    const values = data.values;
    
    // Base configuration
    const baseConfig = {
        type: chartType === 'area' ? 'line' : (chartType === 'horizontalBar' ? 'bar' : chartType),
        data: {
            labels: labels,
            datasets: [{
                label: 'Data',
                data: values
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            plugins: {
                tooltip: {
                    backgroundColor: 'rgba(0, 0, 0, 0.8)',
                    padding: 12,
                    titleFont: { size: 14 },
                    bodyFont: { size: 13 }
                }
            }
        }
    };
    
    // Configure based on chart type
    if (chartType === 'bar') {
        baseConfig.data.datasets[0].backgroundColor = 'rgba(102, 126, 234, 0.6)';
        baseConfig.data.datasets[0].borderColor = 'rgba(102, 126, 234, 1)';
        baseConfig.data.datasets[0].borderWidth = 2;
        baseConfig.data.datasets[0].borderRadius = 6;
        baseConfig.options.plugins.legend = { display: false };
        baseConfig.options.scales = {
            y: {
                beginAtZero: true,
                grid: { color: 'rgba(0, 0, 0, 0.05)' }
            },
            x: {
                grid: { display: false }
            }
        };
    }
    else if (chartType === 'horizontalBar') {
        baseConfig.options.indexAxis = 'y';
        baseConfig.data.datasets[0].backgroundColor = 'rgba(102, 126, 234, 0.6)';
        baseConfig.data.datasets[0].borderColor = 'rgba(102, 126, 234, 1)';
        baseConfig.data.datasets[0].borderWidth = 2;
        baseConfig.data.datasets[0].borderRadius = 6;
        baseConfig.options.plugins.legend = { display: false };
        baseConfig.options.scales = {
            x: {
                beginAtZero: true,
                grid: { color: 'rgba(0, 0, 0, 0.05)' }
            },
            y: {
                grid: { display: false }
            }
        };
    }
    else if (chartType === 'line') {
        baseConfig.data.datasets[0].backgroundColor = 'rgba(102, 126, 234, 0.2)';
        baseConfig.data.datasets[0].borderColor = 'rgba(102, 126, 234, 1)';
        baseConfig.data.datasets[0].borderWidth = 3;
        baseConfig.data.datasets[0].fill = false;
        baseConfig.data.datasets[0].tension = 0.3;
        baseConfig.data.datasets[0].pointBackgroundColor = 'rgba(102, 126, 234, 1)';
        baseConfig.data.datasets[0].pointBorderColor = '#fff';
        baseConfig.data.datasets[0].pointBorderWidth = 2;
        baseConfig.data.datasets[0].pointRadius = 5;
        baseConfig.options.plugins.legend = { display: false };
        baseConfig.options.scales = {
            y: {
                beginAtZero: true,
                grid: { color: 'rgba(0, 0, 0, 0.05)' }
            },
            x: {
                grid: { display: false }
            }
        };
    }
    else if (chartType === 'area') {
        baseConfig.data.datasets[0].backgroundColor = 'rgba(102, 126, 234, 0.3)';
        baseConfig.data.datasets[0].borderColor = 'rgba(102, 126, 234, 1)';
        baseConfig.data.datasets[0].borderWidth = 3;
        baseConfig.data.datasets[0].fill = true;
        baseConfig.data.datasets[0].tension = 0.3;
        baseConfig.data.datasets[0].pointBackgroundColor = 'rgba(102, 126, 234, 1)';
        baseConfig.data.datasets[0].pointBorderColor = '#fff';
        baseConfig.data.datasets[0].pointBorderWidth = 2;
        baseConfig.data.datasets[0].pointRadius = 5;
        baseConfig.options.plugins.legend = { display: false };
        baseConfig.options.scales = {
            y: {
                beginAtZero: true,
                grid: { color: 'rgba(0, 0, 0, 0.05)' }
            },
            x: {
                grid: { display: false }
            }
        };
    }
    else if (chartType === 'pie' || chartType === 'doughnut') {
        // Generate colors for pie/doughnut charts
        const colors = generateColors(values.length);
        baseConfig.data.datasets[0].backgroundColor = colors.backgrounds;
        baseConfig.data.datasets[0].borderColor = colors.borders;
        baseConfig.data.datasets[0].borderWidth = 2;
        baseConfig.options.plugins.legend = {
            display: true,
            position: 'right',
            labels: {
                padding: 15,
                font: { size: 12 }
            }
        };
    }
    
    return baseConfig;
}

/**
 * Generate colors for multiple data points
 * @param {number} count - Number of colors needed
 * @returns {Object} Object with background and border colors
 */
function generateColors(count) {
    const baseColors = [
        { bg: 'rgba(102, 126, 234, 0.7)', border: 'rgba(102, 126, 234, 1)' },
        { bg: 'rgba(118, 75, 162, 0.7)', border: 'rgba(118, 75, 162, 1)' },
        { bg: 'rgba(255, 99, 132, 0.7)', border: 'rgba(255, 99, 132, 1)' },
        { bg: 'rgba(54, 162, 235, 0.7)', border: 'rgba(54, 162, 235, 1)' },
        { bg: 'rgba(255, 206, 86, 0.7)', border: 'rgba(255, 206, 86, 1)' },
        { bg: 'rgba(75, 192, 192, 0.7)', border: 'rgba(75, 192, 192, 1)' },
        { bg: 'rgba(153, 102, 255, 0.7)', border: 'rgba(153, 102, 255, 1)' },
        { bg: 'rgba(255, 159, 64, 0.7)', border: 'rgba(255, 159, 64, 1)' },
        { bg: 'rgba(201, 203, 207, 0.7)', border: 'rgba(201, 203, 207, 1)' },
        { bg: 'rgba(83, 102, 255, 0.7)', border: 'rgba(83, 102, 255, 1)' }
    ];
    
    const backgrounds = [];
    const borders = [];
    
    for (let i = 0; i < count; i++) {
        const color = baseColors[i % baseColors.length];
        backgrounds.push(color.bg);
        borders.push(color.border);
    }
    
    return { backgrounds, borders };
}

/**
 * Clear the current chart
 */
function clearChart() {
    const chartCard = document.getElementById('chartCard');
    
    if (currentChart) {
        currentChart.destroy();
        currentChart = null;
    }
    
    currentChartData = null;
    
    // Hide the chart card
    chartCard.style.display = 'none';
}

// ====================================
// Utility Functions
// ====================================

/**
 * Show success message
 * @param {string} message - Success message to display
 */
function showSuccess(message) {
    showNotification(message, 'success');
}

/**
 * Show error message
 * @param {string} message - Error message to display
 */
function showError(message) {
    showNotification(message, 'error');
}

/**
 * Show notification message
 * @param {string} message - Message to display
 * @param {string} type - Type of notification (success, error, info)
 */
function showNotification(message, type = 'info') {
    // Create notification element
    const notification = document.createElement('div');
    notification.className = `status-message ${type}`;
    notification.textContent = message;
    
    // Insert at top of content area
    const content = document.querySelector('.content');
    const firstChild = content.firstChild;
    content.insertBefore(notification, firstChild);
    
    // Auto remove after 5 seconds
    setTimeout(() => {
        notification.style.opacity = '0';
        notification.style.transform = 'translateY(-10px)';
        notification.style.transition = 'all 0.3s ease';
        
        setTimeout(() => {
            notification.remove();
        }, 300);
    }, 5000);
}

/**
 * Log message with timestamp
 * @param {string} message - Message to log
 */
function log(message) {
    console.log(`[${new Date().toLocaleTimeString()}] ${message}`);
}

/**
 * Escape HTML to prevent XSS attacks
 * @param {string} text - Text to escape
 * @returns {string} Escaped text
 */
function escapeHtml(text) {
    const div = document.createElement('div');
    div.textContent = text;
    return div.innerHTML;
}

// ====================================
// Initialize
// ====================================

// Add collapsible functionality for Code section
const codeHeader = document.getElementById('codeHeader');
const codeContent = document.getElementById('codeContent');

if (codeHeader && codeContent) {
    codeHeader.addEventListener('click', () => {
        codeHeader.classList.toggle('collapsed');
        codeContent.classList.toggle('collapsed');
    });
}

// Add collapsible functionality for Transparency section
const transparencyHeader = document.getElementById('transparencyHeader');
const transparencyContent = document.getElementById('transparencyContent');

if (transparencyHeader && transparencyContent) {
    transparencyHeader.addEventListener('click', () => {
        transparencyHeader.classList.toggle('collapsed');
        transparencyContent.classList.toggle('collapsed');
    });
}

// Add chart type selector event listener
const chartTypeSelector = document.getElementById('chartTypeSelector');
if (chartTypeSelector) {
    chartTypeSelector.addEventListener('change', (e) => {
        const selectedType = e.target.value;
        renderChartWithType(selectedType);
        log(`Chart type changed to: ${selectedType}`);
    });
}

log('Conversational Data Intelligence Platform initialized');
