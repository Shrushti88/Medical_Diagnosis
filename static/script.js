document.addEventListener('DOMContentLoaded', () => {
    const form = document.getElementById('prediction-form');
    const resultsContainer = document.getElementById('results-container');
    const predictBtn = document.getElementById('predict-btn');
    const predictedDiseaseEl = document.getElementById('predicted-disease');
    const confidenceBadge = document.getElementById('confidence-badge');
    const riskBadge = document.getElementById('risk-badge');
    const explanationText = document.getElementById('explanation-text');
    
    let shapChart = null;

    form.addEventListener('submit', async (e) => {
        e.preventDefault();
        
        // UI Loading State
        predictBtn.textContent = 'Analyzing...';
        predictBtn.disabled = true;
        
        // Gather data
        const formData = new FormData(form);
        const patientData = {
            Age: parseFloat(formData.get('Age')),
            BMI: parseFloat(formData.get('BMI')),
            Glucose: parseFloat(formData.get('Glucose')),
            Blood_Pressure: parseFloat(formData.get('Blood_Pressure')),
            Cholesterol: parseFloat(formData.get('Cholesterol')),
            Heart_Rate: parseFloat(formData.get('Heart_Rate'))
        };

        try {
            // Call API
            const response = await fetch('/predict', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify(patientData)
            });

            if (!response.ok) {
                throw new Error('Prediction request failed');
            }

            const result = await response.json();
            
            // Render Results
            renderResults(result);
            
            // Show results container
            resultsContainer.classList.remove('hidden');
            
            // Scroll to results on mobile
            if (window.innerWidth < 900) {
                resultsContainer.scrollIntoView({ behavior: 'smooth' });
            }
            
        } catch (error) {
            console.error('Error:', error);
            alert('An error occurred while making the prediction.');
        } finally {
            // Restore button state
            predictBtn.textContent = 'Analyze Patient';
            predictBtn.disabled = false;
        }
    });

    function renderResults(result) {
        // Update text
        predictedDiseaseEl.textContent = result.prediction;
        
        // Update badges
        const confPercent = Math.round(result.confidence * 100);
        confidenceBadge.textContent = `Confidence: ${confPercent}%`;
        riskBadge.textContent = `Risk: ${result.risk_level}`;
        
        // Apply color classes
        predictedDiseaseEl.className = '';
        if (result.prediction === 'Healthy') {
            predictedDiseaseEl.classList.add('status-healthy');
            riskBadge.style.backgroundColor = 'rgba(16, 185, 129, 0.2)';
            riskBadge.style.color = '#34d399';
        } else if (result.prediction === 'Diabetes Risk') {
            predictedDiseaseEl.classList.add('status-diabetes');
            riskBadge.style.backgroundColor = 'rgba(245, 158, 11, 0.2)';
            riskBadge.style.color = '#fbbf24';
        } else {
            predictedDiseaseEl.classList.add('status-heart');
            riskBadge.style.backgroundColor = 'rgba(239, 68, 68, 0.2)';
            riskBadge.style.color = '#f87171';
        }
        
        // Update explanation
        explanationText.textContent = result.explanation;
        
        // Render Chart
        renderChart(result.contributions);
    }

    function renderChart(contributions) {
        const ctx = document.getElementById('shapChart').getContext('2d');
        
        // Sort contributions to have largest impact on top
        // SHAP values can be negative (pushing towards Healthy) or positive (pushing towards Disease)
        // Actually, absolute value sort is already done by backend. Let's keep backend sort.
        const labels = contributions.map(c => `${c.feature} (${c.input_value})`);
        const data = contributions.map(c => c.value);
        
        // Colors: green for pushing to Healthy (negative), red for pushing to Disease (positive)
        // Depending on the class! Since it's multiclass, positive means pushing towards THAT class.
        // We'll just use a gradient or simple colors. Positive impact on THIS prediction.
        const backgroundColors = data.map(v => v > 0 ? 'rgba(239, 68, 68, 0.8)' : 'rgba(59, 130, 246, 0.8)');
        
        if (shapChart) {
            shapChart.destroy();
        }

        // Set Chart.js defaults for dark mode
        Chart.defaults.color = '#94a3b8';
        Chart.defaults.font.family = 'Inter';

        shapChart = new Chart(ctx, {
            type: 'bar',
            data: {
                labels: labels,
                datasets: [{
                    label: 'Impact on Prediction',
                    data: data,
                    backgroundColor: backgroundColors,
                    borderRadius: 4,
                }]
            },
            options: {
                indexAxis: 'y', // Horizontal bar chart
                responsive: true,
                maintainAspectRatio: false,
                plugins: {
                    legend: {
                        display: false
                    },
                    tooltip: {
                        callbacks: {
                            label: function(context) {
                                let val = context.raw;
                                return `Impact score: ${val.toFixed(4)}`;
                            }
                        }
                    }
                },
                scales: {
                    x: {
                        grid: {
                            color: 'rgba(255, 255, 255, 0.05)'
                        },
                        title: {
                            display: true,
                            text: 'SHAP Value (Impact)'
                        }
                    },
                    y: {
                        grid: {
                            display: false
                        }
                    }
                }
            }
        });
    }
});
