document.addEventListener("DOMContentLoaded", () => {
  // Activate Feather Icons
  feather.replace();

  // --- Theme Toggler ---
  const themeToggle = document.getElementById("theme-toggle-checkbox");
  const currentTheme = localStorage.getItem("theme");

  if (currentTheme) {
    document.documentElement.setAttribute("data-theme", currentTheme);
    if (currentTheme === "light") {
      themeToggle.checked = false;
    }
  }

  themeToggle.addEventListener("change", function () {
    if (this.checked) {
      document.documentElement.setAttribute("data-theme", "dark");
      localStorage.setItem("theme", "dark");
    } else {
      document.documentElement.setAttribute("data-theme", "light");
      localStorage.setItem("theme", "light");
    }
  });

  // --- Real-time Data Fetching ---
  function updateDashboard() {
    fetch("/dashboard_data")
      .then((res) => res.json())
      .then((data) => {
        // 1. Update Live Workout Summary
        const summary = data.live_summary;
        document.getElementById("summary-reps").innerText = summary.reps;
        document.getElementById("summary-sets").innerText = summary.sets;
        document.getElementById("summary-calories").innerText = summary.calories;

        // Format duration from seconds to M:SS
        const minutes = Math.floor(summary.duration / 60);
        const seconds = summary.duration % 60;
        document.getElementById("summary-duration").innerText = `${minutes}m ${seconds}s`;

        // Update live progress bar (from curl angle)
        document.getElementById("summary-progress-bar").style.width = `${summary.progress}%`;

        // 2. Update Accuracy Ring
        const accuracy = data.accuracy;
        const accuracyRing = document.querySelector(".progress-ring-circle");
        const accuracyText = document.getElementById("accuracy-text");

        accuracyRing.style.setProperty("--value", accuracy);
        accuracyText.innerText = `${accuracy}%`;

      })
      .catch(error => console.error("Error fetching dashboard data:", error));
  }

  // Fetch data every second for real-time updates
  setInterval(updateDashboard, 1000);
  // Initial call to populate data immediately on page load
  updateDashboard();

  // --- Chart.js Implementation ---
  const chartOptions = {
    maintainAspectRatio: false,
    plugins: {
      legend: {
        display: false,
      },
    },
    scales: {
      x: {
        ticks: { color: "#8a8a8a" },
        grid: { color: "rgba(255, 255, 255, 0.1)" },
      },
      y: {
        ticks: { color: "#8a8a8a" },
        grid: { color: "rgba(255, 255, 255, 0.1)" },
        beginAtZero: true,
      },
    },
  };

  function initializeCharts() {
    fetch("/chart_data")
      .then(res => res.json())
      .then(chartData => {
        // --- Weekly Progress Chart (Bar) ---
        const weeklyCtx = document.getElementById("weeklyProgressChart");
        if (weeklyCtx && chartData.weekly_progress) {
          new Chart(weeklyCtx, {
            type: "bar",
            data: {
              labels: chartData.weekly_progress.labels,
              datasets: [
                {
                  label: "Workout Duration (min)",
                  data: chartData.weekly_progress.data,
                  backgroundColor: "#00ccff",
                  borderRadius: 6,
                },
              ],
            },
            options: chartOptions,
          });
        }
      })
      .catch(error => console.error("Error fetching chart data:", error));
  }

  // Load the charts with dynamic data
  initializeCharts();
});