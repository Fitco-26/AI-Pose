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
      if (typeof EXERCISE_ID !== "undefined" && EXERCISE_ID === "squats") {
        // For squats, show only total reps and hide other info.
        repsEl.innerText = `Total Reps: ${data.total}`;
        stageEl.innerText = `Stage: ${data.stage}`; // Show stage
        stageEl.style.display = "block";
        warningEl.style.display = "block"; // Show real-time warning text
        errorLogContainer.style.display = "block"; // Show the form feedback log
      } else {
        // Default behavior for bicep curls
        repsEl.innerText = `Left: ${data.left} | Right: ${data.right} | Total: ${data.total}`;
        stageEl.innerText = `Stage: ${data.stage}`;
        stageEl.style.display = "block";
        warningEl.style.display = "block";
        if (errorLogContainer.style.display !== "none")
          errorLogContainer.style.display = "block";
      }

      warningEl.innerText = data.warning;

      // --- Live Error Log Update ---
      // This section now runs on every update, not just at the end.
      if (data.error_log && data.error_log.length > 0) {
        errorLogEl.innerHTML =
          "<strong>Form Feedback:</strong><br>" + data.error_log.join("<br>");
      } else {
        // If the log is empty but the container is visible (i.e., workout started), show the default message.
        if (!errorLogContainer.classList.contains("hidden")) {
          errorLogEl.innerHTML =
            "<strong>Form Feedback:</strong><br><span class='no-feedback'>No feedback yet. Keep up the great form!</span>";
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
  errorLogEl.innerHTML =
    "<strong>Form Feedback:</strong><br><span class='no-feedback'>No feedback yet. Keep up the great form!</span>";

  // Start the actual workout on the backend
  fetch("/start", { method: "POST" });
}

function hideIntro() {
  // Hide the overlay and stop the video
  introOverlay.classList.add("hidden");
  introVideo.pause();
  introVideo.currentTime = 0; // Reset video for next time
}

async function stopWorkout() {
  try {
    const response = await fetch("/stop", { method: "POST" });
    const data = await response.json();

    if (data.status === "stopped" && data.summary) {
      const s = data.summary;
      showSessionPopup(
        s.exercise || "Workout",
        s.total_reps || 0,
        s.avg_angle || 0,
        s.improvement_percent || 0,
        s.feedback || "Good session!",
        s.explanation || "You did great — keep up the effort!",
        s.form_accuracy,
        s.smoothness_score, // Pass new data
        s.issue_counts // Pass new data
      );
    } else {
      // 🟢 Fallback popup if no proper summary was returned
      showSessionPopup(
        "Workout",
        0,
        0,
        0,
        "Workout stopped successfully.",
        "Session ended — data not saved or exercise handler was inactive."
      );
    }
  } catch (err) {
    console.error("Error stopping exercise:", err);
    alert("Something went wrong while stopping the workout.");
  }
}

function showSessionPopup(
  exercise,
  reps,
  avgAngle,
  improvement,
  feedback,
  explanation,
  formAccuracy, // Added to receive the accurate value from the backend
  smoothnessScore,
  issueCounts
) {
  // --- Form Accuracy Calculation ---
  const targetAngle = exercise.toLowerCase() === "squats" ? 100 : 45; // ideal targets
  let deviation = Math.abs(avgAngle - targetAngle);
  let formScore = Math.max(0, 100 - (deviation / targetAngle) * 100);
  let formColor =
    formScore > 85 ? "#00ff99" : formScore > 70 ? "#ffaa00" : "#ff4d4d";

  formScore = Math.min(formScore, 100);
  formScore += improvement * 0.2; // small influence of improvement%
  formScore = Math.max(0, Math.min(100, formScore));

  // ✅ Use the accurate value from the backend directly
  const finalFormScore = formAccuracy !== undefined ? formAccuracy : formScore;

  // --- Create popup container ---
  const overlay = document.createElement("div");
  overlay.className = "popup-overlay";
  overlay.innerHTML = `
      <div class="popup-container">
          <h2>🏋️ ${exercise.toUpperCase()} Summary</h2>
          <p>Reps: <b>${reps}</b></p>
          <p>Average Angle: <b>${avgAngle.toFixed(1)}°</b></p>
          <p>Improvement: <b>${improvement.toFixed(2)}%</b></p>
          <p>🎯 Form Accuracy: 
            <b style="color:${formColor};">${finalFormScore.toFixed(1)}%</b>
          </p>
          <p class="feedback-text">"${feedback}"</p>

          <div class="popup-button-group">
              <button id="explainSummaryBtn" class="button" style="background: #333; color: #fff;">Explain</button>
              <button id="closeSummaryBtn" class="button" style="background: #00ff99; color: #000;">Close</button>
          </div>
      </div>
  `;
  document.body.appendChild(overlay);

  // Animate in
  setTimeout(() => overlay.classList.add("visible"), 10);

  const closePopup = () => {
    overlay.classList.remove("visible");
    setTimeout(() => overlay.remove(), 300);
  };

  overlay
    .querySelector("#closeSummaryBtn")
    .addEventListener("click", closePopup);
  overlay.querySelector("#explainSummaryBtn").addEventListener("click", () => {
    closePopup();
    showExplanationPopup(
      exercise,
      reps,
      avgAngle,
      improvement,
      feedback,
      explanation,
      finalFormScore, // Pass the corrected score
      smoothnessScore,
      issueCounts
    );
  });
  overlay.addEventListener("click", (e) => {
    if (e.target === overlay) closePopup();
  });
}

// Event Listeners for the intro video and workout controls
startButton.addEventListener("click", startWorkout);
skipIntroButton.addEventListener("click", hideIntro);

// --- Target Reps ---
const targetRepsInput = document.getElementById("targetRepsInput");

targetRepsInput.addEventListener("change", (event) => {
  const newTarget = parseInt(event.target.value, 10);
  if (newTarget > 0) {
    fetch("/set_target_reps", {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
      },
      body: JSON.stringify({ target: newTarget }),
    });
  }
});

// --- Landmark Toggle ---
const landmarksToggle = document.getElementById("landmarksToggle");

landmarksToggle.addEventListener("change", (event) => {
  const showLandmarks = event.target.checked;
  fetch("/toggle_landmarks", {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
    },
    body: JSON.stringify({ show: showLandmarks }),
  });
});

// ----------------------- Layout / Resizer -----------------------
const videoPanel = document.getElementById("videoPanel");
const statsPanel = document.getElementById("statsPanel");
const resizer = document.getElementById("resizer");

let isResizing = false;

resizer.addEventListener("mousedown", (e) => {
  isResizing = true;
  document.body.style.cursor = "col-resize";
  // Disable transitions during drag for immediate feedback
  videoPanel.classList.add("no-transition");
  statsPanel.classList.add("no-transition");
});

document.addEventListener("mousemove", (e) => {
  if (!isResizing) return;

  const container = videoPanel.parentNode;
  const totalWidth = container.offsetWidth;
  const resizerWidth = resizer.offsetWidth;
  const gap = 10; // The gap value from CSS

  const newVideoFlex =
    (e.clientX - container.offsetLeft) / (totalWidth - resizerWidth - gap);
  const newStatsFlex = 1 - newVideoFlex;

  videoPanel.style.flex = newVideoFlex;
  statsPanel.style.flex = newStatsFlex;
});

document.addEventListener("mouseup", (e) => {
  isResizing = false;
  document.body.style.cursor = "default";
  // Re-enable transitions after drag is complete
  videoPanel.classList.remove("no-transition");
  statsPanel.classList.remove("no-transition");
});

// ----------------------- Double-click Fullscreen -----------------------
let fullScreen = false;
videoPanel.addEventListener("dblclick", () => {
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
// ----------------------- Dynamic Explanation Popup -----------------------
function showExplanationPopup(
  exercise,
  reps,
  avgAngle,
  improvement,
  feedback,
  explanation,
  formAccuracy, // Receive the accurate score
  smoothnessScore,
  issueCounts
) {
  const improvementColor =
    improvement > 0 ? "#00ff99" : improvement < 0 ? "#ff6b6b" : "#ffaa00";
  const improvementEmoji =
    improvement > 0 ? "✅" : improvement < 0 ? "⚠️" : "➖";
  const fadeDuration = 300;

  const finalFormScore = formAccuracy; // Use the value passed from the first popup

  const overlay = document.createElement("div");
  overlay.className = "popup-overlay";

  const popup = document.createElement("div");
  popup.className = "popup-container explanation-popup"; // Use classes

  popup.innerHTML = `
      <h2>
        📊 Detailed ${exercise.toUpperCase()} Breakdown
      </h2>

      <!-- Form Accuracy Bar -->
      <div class="accuracy-bar-container">
        <p>🎯 Form Accuracy: ${finalFormScore.toFixed(1)}%</p>
        <div class="accuracy-progress-bg">
            <div class="accuracy-progress-fill" style="width: ${finalFormScore.toFixed(
              1
            )}%; background: linear-gradient(90deg, ${
    finalFormScore > 85
      ? "#00ccff, #00ff99"
      : finalFormScore > 70
      ? "#ffaa00, #ffcc00"
      : "#ff6b6b, #ff4d4d"
  });"></div>
        </div>
      </div>

      <!-- Issue Counts Section (only if issues exist) -->
      ${
        issueCounts && Object.keys(issueCounts).length > 0
          ? `
        <div class="issue-counts-container">
          <h4>Common Issues This Session:</h4>
          <ul>
            ${Object.entries(issueCounts)
              .map(
                ([issue, count]) =>
                  `<li><span>${issue.replace(
                    /_/g,
                    " "
                  )}:</span> <strong>${count} reps</strong></li>`
              )
              .join("")}
          </ul>
        </div>`
          : ""
      }

      <table>
        <thead>
          <tr>
            <th>Metric</th>
            <th>Value</th>
            <th>Explanation</th>
          </tr>
        </thead>
        <tbody>
          <tr>
            <td>🏁 Reps</td>
            <td>${reps}</td>
            <td>You completed ${reps} full ${exercise.toLowerCase()} cycles (down → up transitions).</td>
          </tr>
          <tr>
            <td>📐 Average Angle</td>
            <td>${avgAngle.toFixed(1)}°</td>
            <td>
              ${
                // ✅ Dynamic explanation based on exercise
                exercise.toLowerCase().includes("squat")
                  ? "Measured at the knee joint. 180° = standing, 90° = full squat. Lower = deeper depth."
                  : "Measured at the elbow joint. 180° = arm extended, 45° = full curl. Lower = better contraction."
              }
            </td>
          </tr>
          <tr>
            <td>📈 Improvement</td>
            <td style="color:${improvementColor}; font-weight:600;">
              ${improvementEmoji} ${improvement.toFixed(2)}%
            </td>
            <td>
            ${(() => {
              if (improvement === 0) {
                return "Baseline session — establishing your starting form!";
              } else if (improvement > 0) {
                return `You improved by +${improvement.toFixed(
                  2
                )}%! Great progress!`;
              } else {
                return `Slight dip of ${Math.abs(improvement).toFixed(
                  2
                )}% — focus on control!`;
              }
            })()}
          </td>
          </tr>
          ${
            // Conditionally add the Smoothness row only if the value is present
            smoothnessScore !== null && smoothnessScore !== undefined
              ? `
          <tr>
            <td>🏃‍♂️ Smoothness</td>
            <td>${(smoothnessScore * 100).toFixed(1)}%</td>
            <td>
              A measure of how controlled your movements were. Higher is better, indicating less jerky motion.
            </td>
          </tr>
          `
              : ""
          }
          <tr>
            <td>💬 Feedback</td>
            <td colspan="2" style="font-style: italic; color:#ccc;">"${feedback}"</td>
          </tr>
        </tbody>
      </table>
      <div class="popup-button-group">
        <button id="closeExplain" class="button" style="background: #00ff99; color: #000;">Close</button>
      </div>
  `;

  overlay.appendChild(popup);
  document.body.appendChild(overlay);
  popup.classList.add("explanation-popup"); // Add class for specific styling

  // Fade-in animation
  setTimeout(() => overlay.classList.add("visible"), 10);

  // Close popup on click
  const closePopup = () => {
    overlay.classList.remove("visible");
    setTimeout(() => overlay.remove(), 300); // Use consistent timing
  };

  popup.querySelector("#closeExplain").addEventListener("click", closePopup);
  overlay.addEventListener("click", (e) => {
    // Close if clicking on the dark background, not the popup content
    if (e.target === overlay) {
      closePopup();
    }
  });
}
