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
      document.getElementById("reps").innerText = `Left: ${data.left} | Right: ${
        data.right
      } | Total: ${data.total}`;
      document.getElementById("stage").innerText = `Stage: ${data.stage}`;
      document.getElementById("warning").innerText = data.warning;
      // Set the target for our animation, don't update the bar directly
      targetProgress = data.progress;
    });
}

// Start the animation loop
requestAnimationFrame(animateProgressBar);

// Fetch stats from the server more frequently for a responsive feel
// The animation loop will handle the smoothing
setInterval(updateStats, 250);

function startWorkout() {
    fetch('/start', { method: "POST" });
}
function stopWorkout() {
    fetch('/stop', { method: "POST" });
}

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
