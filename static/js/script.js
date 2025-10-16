// ----------------------- Stats Update -----------------------
const progressBar = document.getElementById("progress");
let targetProgress = 0;
let currentProgress = 0;

// Animation loop for the progress bar
function animateProgressBar() {
  // Smoothly move current progress towards the target
  const easing = 0.4; // Increased easing for a much more responsive feel
  currentProgress += (targetProgress - currentProgress) * easing;

  // Update the actual progress bar value
  progressBar.value = currentProgress;

  // Keep the animation running
  requestAnimationFrame(animateProgressBar);
}

function updateStats() {
  fetch("/stats")
    .then((res) => res.json())
    .then((data) => {
      const errorLogContainer = document.getElementById("errorLogContainer");
      const errorLogEl = document.getElementById("errorLog");
      const warningEl = document.getElementById("warning");
      const stageEl = document.getElementById("stage");
      const repsEl = document.getElementById("reps");

      // --- UI Customization based on Exercise ---
      if (typeof EXERCISE_ID !== 'undefined' && EXERCISE_ID === 'squats') {
        // For squats, show only total reps and hide other info.
        repsEl.innerText = `Total Reps: ${data.total}`;
        stageEl.innerText = `Stage: ${data.stage}`; // Show stage
        stageEl.style.display = 'block';
        warningEl.style.display = 'block'; // Show real-time warning text
        errorLogContainer.style.display = 'block'; // Show the form feedback log
      } else {
        // Default behavior for bicep curls
        repsEl.innerText = `Left: ${data.left} | Right: ${data.right} | Total: ${data.total}`;
        stageEl.innerText = `Stage: ${data.stage}`;
        stageEl.style.display = 'block';
        warningEl.style.display = 'block';
        if (errorLogContainer.style.display !== 'none') errorLogContainer.style.display = 'block';
      }

      warningEl.innerText = data.warning;

      // --- Live Error Log Update ---
      // This section now runs on every update, not just at the end.
      if (data.error_log && data.error_log.length > 0) {
        errorLogEl.innerHTML = "<strong>Form Feedback:</strong><br>" + data.error_log.join("<br>");
      } else {
        // If the log is empty but the container is visible (i.e., workout started), show the default message.
        if (!errorLogContainer.classList.contains("hidden")) {
            errorLogEl.innerHTML = "<strong>Form Feedback:</strong><br><span class='no-feedback'>No feedback yet. Keep up the great form!</span>";
        }
      }

      // Handle workout completion state
      if (data.workout_complete) {
        warningEl.style.color = "#00ff99"; // Success green for the main message
      } else {
        warningEl.style.color = ""; // Revert to default CSS color
      }

      // Set the target for our animation, don't update the bar directly
      targetProgress = data.progress;
    });
}

// Start the animation loop
requestAnimationFrame(animateProgressBar);

// Fetch stats from the server more frequently for a responsive feel
// The animation loop will handle the smoothing
setInterval(updateStats, 250);

// --- Workout Control ---
const introOverlay = document.getElementById("introVideoOverlay");
const introVideo = document.getElementById("introVideo");
const startButton = document.getElementById("startButton");
const skipIntroButton = document.getElementById("skipIntroButton");

function startWorkout() {
    // Hide intro overlay if it's still visible, so the user sees the action start.
    if (!introOverlay.classList.contains("hidden")) {
        hideIntro();
    }

    // Make the error log container visible and set its initial state.
    const errorLogContainer = document.getElementById("errorLogContainer");
    const errorLogEl = document.getElementById("errorLog");
    errorLogContainer.classList.remove("hidden");
    errorLogEl.innerHTML = "<strong>Form Feedback:</strong><br><span class='no-feedback'>No feedback yet. Keep up the great form!</span>";

    // Start the actual workout on the backend
    fetch('/start', { method: "POST" });
}

function hideIntro() {
    // Hide the overlay and stop the video
    introOverlay.classList.add("hidden");
    introVideo.pause();
    introVideo.currentTime = 0; // Reset video for next time
}

function stopWorkout() {
    fetch('/stop', { method: "POST" });
}

// Event Listeners for the intro video and workout controls
startButton.addEventListener('click', startWorkout);
skipIntroButton.addEventListener('click', hideIntro);
introVideo.addEventListener('ended', hideIntro); // Auto-hide when video ends


// --- Target Reps ---
const targetRepsInput = document.getElementById("targetRepsInput");

targetRepsInput.addEventListener('change', (event) => {
    const newTarget = parseInt(event.target.value, 10);
    if (newTarget > 0) {
        fetch('/set_target_reps', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({ target: newTarget }),
        });
    }
});

// ----------------------- Layout / Resizer -----------------------
const videoPanel = document.getElementById("videoPanel");
const statsPanel = document.getElementById("statsPanel");
const resizer = document.getElementById("resizer");

let isResizing = false;

resizer.addEventListener('mousedown', e => {
    isResizing = true;
    document.body.style.cursor = 'col-resize';
    // Disable transitions during drag for immediate feedback
    videoPanel.classList.add('no-transition');
    statsPanel.classList.add('no-transition');
});

document.addEventListener('mousemove', e => {
    if (!isResizing) return;

    const container = videoPanel.parentNode;
    const totalWidth = container.offsetWidth;
    const resizerWidth = resizer.offsetWidth;
    const gap = 10; // The gap value from CSS

    const newVideoFlex = (e.clientX - container.offsetLeft) / (totalWidth - resizerWidth - gap);
    const newStatsFlex = 1 - newVideoFlex;

    videoPanel.style.flex = newVideoFlex;
    statsPanel.style.flex = newStatsFlex;
});

document.addEventListener('mouseup', e => {
    isResizing = false;
    document.body.style.cursor = 'default';
    // Re-enable transitions after drag is complete
    videoPanel.classList.remove('no-transition');
    statsPanel.classList.remove('no-transition');
});

// ----------------------- Double-click Fullscreen -----------------------
let fullScreen = false;
videoPanel.addEventListener('dblclick', () => {
    if (!fullScreen) {
        videoPanel.style.flex = "1 1 100%";
        statsPanel.style.display = "none";
        resizer.style.display = "none";
        fullScreen = true;
    } else {
        videoPanel.style.flex = "7"; // Restore flex ratio
        statsPanel.style.display = "block";
        statsPanel.style.flex = "3"; // Restore flex ratio
        resizer.style.display = "block";
        fullScreen = false;
    }
});
