// Main JavaScript file for Sentiment Analysis App

document.addEventListener('DOMContentLoaded', function() {
    // Add active class to current nav item
    const currentLocation = window.location.pathname;
    const navLinks = document.querySelectorAll('.nav-link');
    
    navLinks.forEach(link => {
        const href = link.getAttribute('href');
        if (href === currentLocation || 
            (href !== '/' && currentLocation.startsWith(href))) {
            link.classList.add('active');
        }
    });

    // Animate stat counters
    const statValues = document.querySelectorAll('.stat-value');
    
    if (statValues.length > 0) {
        statValues.forEach(stat => {
            const finalValue = parseInt(stat.textContent);
            animateCounter(stat, 0, finalValue, 2000);
        });
    }

    // Form submission with loading indicator
    const predictionForm = document.getElementById('prediction-form');
    if (predictionForm) {
        predictionForm.addEventListener('submit', function() {
            const submitBtn = document.querySelector('button[type="submit"]');
            const originalText = submitBtn.innerHTML;
            submitBtn.innerHTML = '<span class="spinner-border spinner-border-sm" role="status" aria-hidden="true"></span> Processing...';
            submitBtn.disabled = true;
            
            // Re-enable after 30 seconds in case of error
            setTimeout(() => {
                submitBtn.innerHTML = originalText;
                submitBtn.disabled = false;
            }, 30000);
        });
    }

    // Visualization form enhancements
    const visualizationForm = document.getElementById('visualization-form');
    if (visualizationForm) {
        const visualizationType = document.getElementById('visualization_type');
        const variationField = document.getElementById('variation-field');
        
        if (visualizationType && variationField) {
            visualizationType.addEventListener('change', function() {
                if (this.value === 'wordcloud') {
                    variationField.style.display = 'block';
                } else {
                    variationField.style.display = 'none';
                }
            });
            
            // Trigger on page load
            if (visualizationType.value === 'wordcloud') {
                variationField.style.display = 'block';
            } else {
                variationField.style.display = 'none';
            }
        }
    }

    // Add parallax effect to background
    window.addEventListener('scroll', function() {
        const scrollPosition = window.pageYOffset;
        document.body.style.backgroundPosition = `0px ${scrollPosition * 0.05}px`;
    });

    // Add animation to cards on scroll
    const animateOnScroll = function() {
        const cards = document.querySelectorAll('.card');
        cards.forEach(card => {
            const cardPosition = card.getBoundingClientRect().top;
            const screenPosition = window.innerHeight / 1.3;
            
            if (cardPosition < screenPosition) {
                card.style.opacity = '1';
                card.style.transform = 'translateY(0)';
            }
        });
    };

    // Set initial state for cards
    const cards = document.querySelectorAll('.card:not(.stat-card)');
    cards.forEach(card => {
        card.style.opacity = '0';
        card.style.transform = 'translateY(20px)';
        card.style.transition = 'all 0.5s ease';
    });

    // Run animation on scroll
    window.addEventListener('scroll', animateOnScroll);
    // Run once on page load
    animateOnScroll();

    // Add tooltip initialization
    const tooltipTriggerList = [].slice.call(document.querySelectorAll('[data-bs-toggle="tooltip"]'));
    tooltipTriggerList.map(function (tooltipTriggerEl) {
        return new bootstrap.Tooltip(tooltipTriggerEl);
    });
});

// Function to animate counters
function animateCounter(element, start, end, duration) {
    let startTimestamp = null;
    const step = (timestamp) => {
        if (!startTimestamp) startTimestamp = timestamp;
        const progress = Math.min((timestamp - startTimestamp) / duration, 1);
        const value = Math.floor(progress * (end - start) + start);
        element.innerHTML = value.toLocaleString();
        if (progress < 1) {
            window.requestAnimationFrame(step);
        }
    };
    window.requestAnimationFrame(step);
}

// Function to show loading spinner
function showLoading(buttonId, loadingText = 'Processing...') {
    const button = document.getElementById(buttonId);
    if (button) {
        button.dataset.originalText = button.innerHTML;
        button.innerHTML = `<span class="spinner-border spinner-border-sm" role="status" aria-hidden="true"></span> ${loadingText}`;
        button.disabled = true;
    }
}

// Function to hide loading spinner
function hideLoading(buttonId) {
    const button = document.getElementById(buttonId);
    if (button && button.dataset.originalText) {
        button.innerHTML = button.dataset.originalText;
        button.disabled = false;
    }
}

// Add particle background effect
function createParticles() {
    const particleContainer = document.createElement('div');
    particleContainer.className = 'particle-container';
    particleContainer.style.position = 'fixed';
    particleContainer.style.top = '0';
    particleContainer.style.left = '0';
    particleContainer.style.width = '100%';
    particleContainer.style.height = '100%';
    particleContainer.style.pointerEvents = 'none';
    particleContainer.style.zIndex = '-1';
    document.body.appendChild(particleContainer);

    for (let i = 0; i < 50; i++) {
        const particle = document.createElement('div');
        particle.className = 'particle';
        particle.style.position = 'absolute';
        particle.style.width = `${Math.random() * 5 + 1}px`;
        particle.style.height = particle.style.width;
        particle.style.background = 'rgba(255, 153, 0, 0.2)';
        particle.style.borderRadius = '50%';
        particle.style.top = `${Math.random() * 100}%`;
        particle.style.left = `${Math.random() * 100}%`;
        particle.style.animation = `float ${Math.random() * 10 + 5}s linear infinite`;
        particle.style.opacity = Math.random() * 0.5;
        particleContainer.appendChild(particle);
    }
}

// Call particle creation on load
window.addEventListener('load', createParticles);