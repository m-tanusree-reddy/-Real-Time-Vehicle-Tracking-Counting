/**
 * Backend URL Configuration
 * --------------------------
 * This section determines the backend API URL to connect to.
 * It prioritizes in the following order:
 * 1. A 'backend' URL parameter (e.g., ?backend=http://192.168.1.10:5000). If found,
 * it's saved to localStorage and the page is reloaded without the parameter.
 * 2. A previously saved URL from localStorage.
 * 3. A default fallback URL (http://127.0.0.1:5000).
 */
const saved = localStorage.getItem("backend_url");
const param = new URL(window.location.href).searchParams.get("backend");
if (param) {
  // If a 'backend' URL parameter exists, save it for future visits.
  localStorage.setItem("backend_url", param);
  const u = new URL(window.location.href);
  u.searchParams.delete("backend"); // Clean the URL
  window.location.replace(u.toString()); // Reload the page with the clean URL
}
const BACKEND_BASE_URL = param || saved || "http://127.0.0.1:5000";

// --- DOM Element References ---
// Get and cache references to all the HTML elements we'll need to interact with.
const videoEl = document.getElementById("video");
const overlayEl = document.getElementById("loadingOverlay");
const statusBar = document.getElementById("statusBar");
const statusText = document.getElementById("statusText");
const upEl = document.getElementById("count-up");
const downEl = document.getElementById("count-down");
const totalEl = document.getElementById("count-total");
const rateUpEl = document.getElementById("rate-up");
const rateDownEl = document.getElementById("rate-down");
const fpsEl = document.getElementById("fps");
const framesEl = document.getElementById("frames");
const lastUpdatedEl = document.getElementById("last-updated");
const backendUrlEl = document.getElementById("backendUrl");
const btnReset = document.getElementById("btnReset");
const btnPause = document.getElementById("btnPause");
const btnDownload = document.getElementById("btnDownload");

// --- Constants and State Variables ---
// Configuration for update intervals.
const UPDATE_INTERVAL_MS = 1000; // How often to fetch new counts (1 second).
const HEALTH_INTERVAL_MS = 30000; // How often to check if the backend is alive (30 seconds).
const METRICS_INTERVAL_MS = 2000; // How often to fetch performance metrics like FPS (2 seconds).
const HISTORY_LIMIT = 900; // Maximum number of data points to store for the CSV download.

// Application state variables.
let isPaused = false; // Toggles whether the app is polling for new data.
let countsInFlight = false; // Flag to prevent multiple simultaneous requests for counts.
let metricsInFlight = false; // Flag for metrics requests.
let healthInFlight = false; // Flag for health check requests.
let previousCounts = { up: 0, down: 0 }; // Store the last known counts to detect changes.
let startTime = Date.now(); // Timestamp for when the app started, used for calculating rates.
let countHistory = []; // Array to store historical data for CSV export.

// --- Initial Setup ---
// Display the determined backend URL in the footer.
backendUrlEl.textContent = BACKEND_BASE_URL;
// Set the source of the video element to the backend's video feed endpoint.
videoEl.src = `${BACKEND_BASE_URL}/video_feed`;

/**
 * Updates the UI status indicator.
 * @param {boolean} ok - If true, the status is 'ok' (green). If false, it's 'error' (red).
 * @param {string} msg - The message to display (e.g., "Live", "Stream error").
 */
function setStatus(ok, msg) {
  statusText.textContent = msg;
  statusBar.classList.toggle("ok", !!ok);
  statusBar.classList.toggle("error", !ok);
}

// --- Video Element Event Handlers ---
// When the video stream loads successfully, hide the loading overlay and set status to "Live".
videoEl.onload = () => { overlayEl.style.display="none"; setStatus(true,"Live"); };
// If there's an error loading the video, show the overlay and an error message.
videoEl.onerror = () => { overlayEl.style.display="flex"; setStatus(false,"Stream error — check backend"); };

/**
 * Fetches and updates the vehicle counts from the backend API.
 */
async function refreshCounts() {
  // Don't fetch if paused or if another request is already in progress.
  if (isPaused || countsInFlight) return;
  countsInFlight = true;

  try {
    const res = await fetch(`${BACKEND_BASE_URL}/api/counts`, { cache: "no-store" });
    if (!res.ok) throw new Error(`HTTP ${res.status}`);
    const data = await res.json();
    
    // Update UI with the new counts.
    const up = Number(data.up ?? 0);
    const down = Number(data.down ?? 0);
    const total = up + down;
    animateValue(upEl, up, previousCounts.up);
    animateValue(downEl, down, previousCounts.down);
    totalEl.textContent = total.toString();

    // Calculate and display the rate (vehicles per minute).
    const elapsedMin = (Date.now() - startTime) / 60000;
    if (elapsedMin > 0) {
      rateUpEl.textContent = `${(up / elapsedMin).toFixed(1)} / min`;
      rateDownEl.textContent = `${(down / elapsedMin).toFixed(1)} / min`;
    }
    
    // Store history for CSV download and manage its size.
    countHistory.push({ ts: new Date().toISOString(), up, down, total });
    if (countHistory.length > HISTORY_LIMIT) {
      countHistory.splice(0, countHistory.length - HISTORY_LIMIT);
    }
    
    // Update state for the next cycle.
    previousCounts = { up, down };
    lastUpdatedEl.textContent = `Last updated: ${new Date().toLocaleTimeString()}`;
    if (statusBar.classList.contains("error")) setStatus(true, "Live"); // Restore status if it was in error.
  } catch (err) {
    console.error("Counts fetch failed:", err);
    setStatus(false, "Backend unreachable");
  } finally {
    countsInFlight = false; // Always reset the flag.
  }
}

/**
 * Fetches and updates performance metrics like FPS from the backend.
 */
async function refreshMetrics() {
  if (metricsInFlight) return;
  metricsInFlight = true;
  try {
    const res = await fetch(`${BACKEND_BASE_URL}/api/metrics`, { cache: "no-store" });
    if (!res.ok) return;
    const m = await res.json();
    fpsEl.textContent = (m.fps ?? 0).toString();
    framesEl.textContent = `${m.frames_processed ?? 0} frames`;
  } catch {} // Fail silently if metrics aren't available.
  finally {
    metricsInFlight = false;
  }
}

/**
 * Performs a health check to see if the backend is responsive.
 */
async function healthCheck() {
  if (healthInFlight) return;
  healthInFlight = true;
  try {
    const res = await fetch(`${BACKEND_BASE_URL}/health`, { cache: "no-store" });
    if (!res.ok) throw new Error(`HTTP ${res.status}`);
    const data = await res.json();
    if (data && data.ok) {
      if (statusBar.classList.contains("error")) setStatus(true, "Live");
    } else {
      setStatus(false, "Health check failed");
    }
  } catch (e) {
    setStatus(false, "Backend not responding");
  } finally {
    healthInFlight = false;
  }
}

// --- Polling Intervals ---
// Set up timers to repeatedly call the fetch functions.
setInterval(refreshCounts, UPDATE_INTERVAL_MS);
setInterval(refreshMetrics, METRICS_INTERVAL_MS);
setInterval(healthCheck, HEALTH_INTERVAL_MS);

// Immediately call the functions once on page load to populate data.
refreshCounts();
refreshMetrics();
healthCheck();

// --- Button Event Listeners ---
// Handle the "Reset Counts" button click.
btnReset.addEventListener("click", async () => {
  try {
    const res = await fetch(`${BACKEND_BASE_URL}/api/reset`);
    if (!res.ok) throw new Error(`HTTP ${res.status}`);
    // Reset front-end state and UI to match the backend.
    previousCounts = { up: 0, down: 0 };
    startTime = Date.now();
    countHistory = [];
    upEl.textContent="0"; downEl.textContent="0"; totalEl.textContent="0";
    rateUpEl.textContent="0.0 / min"; rateDownEl.textContent="0.0 / min";
    setStatus(true, "Counts reset");
  } catch (e) {
    setStatus(false, "Reset failed");
  }
});

// Handle the "Pause" / "Resume" button click.
btnPause.addEventListener("click", () => {
  isPaused = !isPaused;
  btnPause.textContent = isPaused ? "Resume" : "Pause";
  setStatus(true, isPaused ? "Paused (polling stopped)" : "Live");
  // If resuming, immediately fetch the latest data.
  if (!isPaused) {
    refreshCounts();
    refreshMetrics();
  }
});

// Handle the "Download CSV" button click.
btnDownload.addEventListener("click", () => {
  if (!countHistory.length) {
    // Using a custom modal or message would be better than alert in a real app.
    // For this PoC, alert is simple and effective.
    alert("No data to download yet.");
    return;
  }
  // Convert the history array to a CSV string.
  let csv = "Timestamp,Up,Down,Total\n";
  countHistory.forEach(r => { csv += `${r.ts},${r.up},${r.down},${r.total}\n`; });
  
  // Create a blob and trigger a download.
  const blob = new Blob([csv], { type: "text/csv" });
  const url = URL.createObjectURL(blob);
  const a = document.createElement("a");
  a.href = url;
  a.download = `vehicle_counts_${new Date().toISOString()}.csv`;
  a.click(); // Programmatically click the link to start the download.
  URL.revokeObjectURL(url); // Clean up the created URL.
});

/**
 * Animates a number change in an element by briefly scaling it up.
 * @param {HTMLElement} el - The element to animate.
 * @param {number} next - The new value.
 * @param {number} prev - The previous value.
 */
function animateValue(el, next, prev) {
  // Only animate if the value has actually changed.
  if (next !== prev) {
    el.textContent = next.toString();
    el.classList.add("pulse");
    setTimeout(() => el.classList.remove("pulse"), 250); // Remove class after animation.
  }
}

// --- Dynamic CSS Injection ---
// Inject the 'pulse' animation style directly into the document's head.
// This keeps the JavaScript self-contained without needing a separate CSS file for this small effect.
const style = document.createElement("style");
style.textContent = `.pulse { transform: scale(1.04); transition: transform .2s; }`;
document.head.appendChild(style);
