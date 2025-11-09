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

async function stopWorkout() {
    try {
        const response = await fetch('/stop', { method: 'POST' });
        const data = await response.json();

        if (data.status === 'stopped' && data.summary) {
            const s = data.summary;
            showSessionPopup(s.exercise, s.total_reps, s.avg_angle, s.improvement_percent, s.feedback);
        } else {
            console.log("Workout stopped, but no summary was returned.");
            // Optionally, just redirect or show a simple message
            // alert("Workout stopped.");
        }
    } catch (err) {
        console.error("Error stopping exercise:", err);
        alert("Something went wrong while stopping the workout.");
    }
}

function showSessionPopup(exercise, reps, avgAngle, improvement, feedback) {
    const popup = document.createElement("div");
    popup.innerHTML = `
        <div style="
            position: fixed; top: 50%; left: 50%; transform: translate(-50%, -50%);
            background: #1f1f1f; color: #fff; border-radius: 12px;
            padding: 24px 32px; text-align: center;
            box-shadow: 0 0 25px rgba(0,0,0,0.4); z-index: 9999;
            width: 90%; max-width: 400px; border: 1px solid rgba(255,255,255,0.1);
        ">
            <h2 style="margin-bottom: 15px; font-size: 22px; color: #00ff99;">🏋️ ${exercise.toUpperCase()} Summary</h2>
            <p style="margin: 8px 0; font-size: 1.1rem;">Reps: <b style="color: #fff;">${reps}</b></p>
            <p style="margin: 8px 0; font-size: 1.1rem;">Average Angle: <b style="color: #fff;">${avgAngle.toFixed(1)}°</b></p>
            <p style="margin: 8px 0; font-size: 1.1rem;">Improvement: <b style="color: #fff;">${improvement.toFixed(2)}%</b></p>
            <p style="margin-top: 15px; font-style: italic; color: #ccc;">"${feedback}"</p>
            <button onclick="this.parentElement.parentElement.remove()" style="
                margin-top: 20px; background: #00ff99; color: #000;
                border: none; padding: 12px 20px; border-radius: 8px;
                cursor: pointer; font-size: 16px; font-weight: 600;
            ">Close</button>
        </div>
    `;
    document.body.appendChild(popup);
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

// --- Landmark Toggle ---
const landmarksToggle = document.getElementById("landmarksToggle");

landmarksToggle.addEventListener('change', (event) => {
    const showLandmarks = event.target.checked;
    fetch('/toggle_landmarks', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json',
        },
        body: JSON.stringify({ show: showLandmarks }),
    });
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
