"use strict";

/* ==========================================================================
   Coffee Doctor — frontend application (vanilla JS, no build step)
   ========================================================================== */

const API_BASE = ""; // same-origin: backend serves this frontend directly
const STORAGE_KEYS = {
  lang: "cd_lang",
  theme: "cd_theme",
  history: "cd_history",
};
const SUPPORTED_LANGS = ["en", "am", "ti", "om"];
const MAX_HISTORY = 20;

const SEVERITY_EMOJI = {
  none: "✅",
  low: "🟢",
  moderate: "🟡",
  high: "🟠",
  critical: "🔴",
};

const CLASS_EMOJI = {
  not_coffee_leaf: "❓",
  healthy: "🌿",
  phoma: "🍂",
  leaf_rust: "🍁",
  cercospora: "🎯",
};

const state = {
  lang: localStorage.getItem(STORAGE_KEYS.lang) || detectBrowserLang(),
  strings: {},
  currentFile: null,
  currentPreviewUrl: null,
  cameraStream: null,
  cameraFacing: "environment",
  libraryCache: {},
  speaking: false,
};

function detectBrowserLang() {
  const nav = (navigator.language || "en").slice(0, 2);
  return SUPPORTED_LANGS.includes(nav) ? nav : "en";
}

/* ---------------------------------------------------------------------- */
/* i18n                                                                    */
/* ---------------------------------------------------------------------- */

async function loadLocale(lang) {
  const res = await fetch(`static/locales/${lang}.json`, { cache: "force-cache" });
  if (!res.ok) throw new Error(`locale ${lang} not found`);
  return res.json();
}

function t(key) {
  return state.strings[key] || key;
}

function applyTranslations() {
  document.documentElement.lang = state.lang;
  document.querySelectorAll("[data-i18n]").forEach((el) => {
    const key = el.getAttribute("data-i18n");
    el.textContent = t(key);
  });
  document.querySelectorAll("[data-i18n-placeholder]").forEach((el) => {
    el.setAttribute("placeholder", t(el.getAttribute("data-i18n-placeholder")));
  });
  document.title = `${t("app_name")} — ${t("app_tagline")}`;
}

async function setLanguage(lang) {
  if (!SUPPORTED_LANGS.includes(lang)) lang = "en";
  state.lang = lang;
  localStorage.setItem(STORAGE_KEYS.lang, lang);
  state.strings = await loadLocale(lang);
  applyTranslations();
  document.getElementById("lang-select").value = lang;
  // Refresh whatever view is showing so translated labels propagate.
  if (document.getElementById("view-library").classList.contains("active")) {
    renderLibrary();
  }
  if (document.getElementById("view-history").classList.contains("active")) {
    renderHistory();
  }
}

/* ---------------------------------------------------------------------- */
/* Theme                                                                   */
/* ---------------------------------------------------------------------- */

function initTheme() {
  const saved = localStorage.getItem(STORAGE_KEYS.theme);
  if (saved) document.documentElement.setAttribute("data-theme", saved);
  updateThemeIcon();
}

function toggleTheme() {
  const current = document.documentElement.getAttribute("data-theme") ||
    (window.matchMedia("(prefers-color-scheme: dark)").matches ? "dark" : "light");
  const next = current === "dark" ? "light" : "dark";
  document.documentElement.setAttribute("data-theme", next);
  localStorage.setItem(STORAGE_KEYS.theme, next);
  updateThemeIcon();
}

function updateThemeIcon() {
  const current = document.documentElement.getAttribute("data-theme") ||
    (window.matchMedia("(prefers-color-scheme: dark)").matches ? "dark" : "light");
  document.getElementById("theme-toggle").textContent = current === "dark" ? "☀️" : "🌙";
}

/* ---------------------------------------------------------------------- */
/* Navigation / views                                                      */
/* ---------------------------------------------------------------------- */

function showView(name) {
  document.querySelectorAll(".view").forEach((v) => v.classList.remove("active"));
  document.getElementById(`view-${name}`).classList.add("active");
  document.querySelectorAll(".bottom-nav button").forEach((b) => {
    b.classList.toggle("active", b.dataset.view === name);
  });
  window.scrollTo({ top: 0, behavior: "instant" in window ? "instant" : "auto" });
  if (name === "library") renderLibrary();
  if (name === "history") renderHistory();
}

/* ---------------------------------------------------------------------- */
/* File handling (upload + camera share one entry point)                   */
/* ---------------------------------------------------------------------- */

const ALLOWED_TYPES = ["image/jpeg", "image/jpg", "image/png", "image/webp"];

function handleFile(file) {
  if (!file) return;
  if (!ALLOWED_TYPES.includes(file.type)) {
    showToast(t("error_file_type"));
    return;
  }
  if (state.currentPreviewUrl) URL.revokeObjectURL(state.currentPreviewUrl);
  state.currentFile = file;
  state.currentPreviewUrl = URL.createObjectURL(file);

  document.getElementById("upload-zone").style.display = "none";
  document.getElementById("tips-card").style.display = "none";
  const previewWrap = document.getElementById("preview-wrap");
  previewWrap.style.display = "block";
  document.getElementById("preview-img").src = state.currentPreviewUrl;
  document.getElementById("analyze-btn").disabled = false;
  document.getElementById("result-card").style.display = "none";
}

function resetUpload() {
  state.currentFile = null;
  if (state.currentPreviewUrl) URL.revokeObjectURL(state.currentPreviewUrl);
  state.currentPreviewUrl = null;
  document.getElementById("upload-zone").style.display = "block";
  document.getElementById("tips-card").style.display = "block";
  document.getElementById("preview-wrap").style.display = "none";
  document.getElementById("analyze-btn").disabled = true;
  document.getElementById("result-card").style.display = "none";
  document.getElementById("file-input").value = "";
}

/* ---------------------------------------------------------------------- */
/* Camera capture                                                          */
/* ---------------------------------------------------------------------- */

async function openCamera() {
  const modal = document.getElementById("camera-modal");
  modal.classList.add("show");
  try {
    const constraints = { video: { facingMode: state.cameraFacing }, audio: false };
    state.cameraStream = await navigator.mediaDevices.getUserMedia(constraints);
    const video = document.getElementById("camera-video");
    video.srcObject = state.cameraStream;
    await video.play();
  } catch (err) {
    closeCamera();
    showToast(t("camera_permission_denied"));
    document.getElementById("camera-fallback-input").click();
  }
}

function closeCamera() {
  const modal = document.getElementById("camera-modal");
  modal.classList.remove("show");
  if (state.cameraStream) {
    state.cameraStream.getTracks().forEach((tr) => tr.stop());
    state.cameraStream = null;
  }
}

function switchCamera() {
  state.cameraFacing = state.cameraFacing === "environment" ? "user" : "environment";
  closeCamera();
  openCamera();
}

function captureFromCamera() {
  const video = document.getElementById("camera-video");
  const canvas = document.createElement("canvas");
  canvas.width = video.videoWidth;
  canvas.height = video.videoHeight;
  canvas.getContext("2d").drawImage(video, 0, 0);
  canvas.toBlob(
    (blob) => {
      const file = new File([blob], `capture-${Date.now()}.jpg`, { type: "image/jpeg" });
      handleFile(file);
      closeCamera();
    },
    "image/jpeg",
    0.92
  );
}

/* ---------------------------------------------------------------------- */
/* Analyze                                                                 */
/* ---------------------------------------------------------------------- */

async function analyze() {
  if (!state.currentFile) return;
  const btn = document.getElementById("analyze-btn");
  const originalLabel = btn.innerHTML;
  btn.disabled = true;
  btn.innerHTML = `<span class="spinner"></span> ${t("btn_analyzing")}`;

  try {
    const form = new FormData();
    form.append("image", state.currentFile);
    form.append("lang", state.lang);

    const res = await fetch(`${API_BASE}/api/v1/predict`, { method: "POST", body: form });
    if (!res.ok) {
      const errBody = await res.json().catch(() => ({}));
      throw new Error(errBody.detail || `HTTP ${res.status}`);
    }
    const data = await res.json();
    renderReport(data, { fromCache: false });
    saveToHistory(data);
  } catch (err) {
    console.error(err);
    showToast(navigator.onLine ? t("error_generic") : t("error_network"));
  } finally {
    btn.disabled = false;
    btn.innerHTML = originalLabel;
  }
}

/* ---------------------------------------------------------------------- */
/* Rendering the report                                                    */
/* ---------------------------------------------------------------------- */

function severityLabel(sev) {
  return t(`severity_${sev}`) || sev;
}

function listSection(id, key, items) {
  if (!items || items.length === 0) return "";
  const lis = items.map((i) => `<li>${escapeHtml(i)}</li>`).join("");
  return `<div class="report-section"><h3>${sectionIcon(key)} ${t(`section_${key}`)}</h3><ul>${lis}</ul></div>`;
}

function textSection(key, text) {
  if (!text) return "";
  return `<div class="report-section"><h3>${sectionIcon(key)} ${t(`section_${key}`)}</h3><p>${escapeHtml(text)}</p></div>`;
}

function sectionIcon(key) {
  const icons = {
    overview: "📋", symptoms: "🔍", causes: "🧬", conditions: "🌦️",
    immediate: "⚡", organic: "🌱", chemical: "🧪", prevention: "🛡️",
    economic: "💰", help: "📞",
  };
  return icons[key] || "•";
}

function escapeHtml(str) {
  const div = document.createElement("div");
  div.textContent = str;
  return div.innerHTML;
}

function probList(labelKey, probs) {
  if (!probs) return "";
  const rows = Object.entries(probs)
    .sort((a, b) => b[1] - a[1])
    .map(([k, v]) => `<li><span>${escapeHtml(prettyClassName(k))}</span><span>${(v * 100).toFixed(1)}%</span></li>`)
    .join("");
  return `<div class="report-section"><h3>${t(labelKey)}</h3><ul class="prob-list">${rows}</ul></div>`;
}

function prettyClassName(key) {
  const map = {
    coffee_leaf: "Coffee leaf", not_coffee_leaf: "Not a coffee leaf",
    healthy: "Healthy", phoma: "Phoma", leaf_rust: "Leaf rust", cercospora: "Cercospora",
  };
  return map[key] || key;
}

function renderReport(data, opts = {}) {
  const report = data.report;
  const card = document.getElementById("result-card");
  card.style.display = "block";

  const emoji = CLASS_EMOJI[data.class_key] || "🌿";
  const isLeaf = data.detector.is_coffee_leaf;

  let banners = "";
  if (data.demo_mode) {
    banners += `<div class="banner banner-warn">🧪 ${t("demo_mode_banner")}</div>`;
  }
  if (data.notice && !data.demo_mode) {
    banners += `<div class="banner banner-warn">⚠️ ${escapeHtml(data.notice)}</div>`;
  }

  let body = "";
  if (!isLeaf) {
    body += `
      <div class="result-header">
        <span class="emoji">${emoji}</span>
        <div>
          <h2>${t("result_not_leaf")}</h2>
          <p class="local-name">${t("result_not_leaf_hint")}</p>
        </div>
      </div>`;
  } else {
    const sev = report.severity || "none";
    body += `
      <div class="result-header">
        <span class="emoji">${emoji}</span>
        <div>
          <h2>${escapeHtml(report.name)}</h2>
          ${report.local_name ? `<p class="local-name">${escapeHtml(report.local_name)}</p>` : ""}
        </div>
      </div>
      <div style="margin:10px 0;">
        <span class="badge badge-${sev}">${SEVERITY_EMOJI[sev] || ""} ${t("result_severity")}: ${severityLabel(sev)}</span>
      </div>
      <div>
        <small>${t("result_confidence")}: ${(data.disease ? data.disease.confidence * 100 : 0).toFixed(1)}%</small>
        <div class="meter"><div class="meter-fill" style="width:${((data.disease ? data.disease.confidence : 0) * 100).toFixed(0)}%"></div></div>
      </div>`;
  }

  body += banners;

  if (isLeaf) {
    body += textSection("overview", report.overview);
    body += listSection("symptoms", "symptoms", report.symptoms);
    body += textSection("causes", report.causes);
    body += listSection("conditions", "conditions", report.favorable_conditions);
    body += listSection("immediate", "immediate", report.immediate_actions);
    body += listSection("organic", "organic", report.organic_treatment);
    body += listSection("chemical", "chemical", report.chemical_treatment);
    body += listSection("prevention", "prevention", report.prevention);
    body += textSection("economic", report.economic_note);
    body += textSection("help", report.when_to_seek_help);
    body += probList("disease_probabilities", data.disease ? data.disease.probabilities : null);
  } else {
    body += listSection("immediate", "immediate", report.immediate_actions);
    body += listSection("prevention", "prevention", report.prevention);
  }

  body += probList("detector_probabilities", data.detector.probabilities);

  body += `
    <div class="report-actions">
      <button class="btn btn-secondary" id="btn-print"><span>🖨️</span> ${t("btn_print_report")}</button>
      <button class="btn btn-secondary" id="btn-download"><span>⬇️</span> ${t("btn_download_report")}</button>
      <button class="btn btn-secondary" id="btn-listen"><span>🔊</span> ${t("btn_listen")}</button>
      <button class="btn btn-secondary" id="btn-share"><span>📤</span> ${t("btn_share")}</button>
    </div>
    <p class="meta-line">${t("request_id_label")}: ${data.request_id || "—"}</p>
  `;

  document.getElementById("result-body").innerHTML = body;
  card.scrollIntoView({ behavior: "smooth", block: "start" });

  document.getElementById("btn-print").addEventListener("click", () => window.print());
  document.getElementById("btn-download").addEventListener("click", () => window.print());
  document.getElementById("btn-share").addEventListener("click", () => shareReport(report));
  document.getElementById("btn-listen").addEventListener("click", () => toggleSpeak(report, isLeaf));

  window._lastReport = data;
}

/* ---------------------------------------------------------------------- */
/* Share + speech                                                          */
/* ---------------------------------------------------------------------- */

function reportToText(report, isLeaf) {
  if (!isLeaf) return `${report.name}. ${report.overview || ""}`;
  const parts = [
    report.name, report.overview, report.causes,
    ...(report.symptoms || []), ...(report.immediate_actions || []),
    ...(report.organic_treatment || []), report.economic_note, report.when_to_seek_help,
  ].filter(Boolean);
  return parts.join(". ");
}

async function shareReport(report) {
  const text = `${report.name} — ${t("app_name")}`;
  if (navigator.share) {
    try {
      await navigator.share({ title: report.name, text, url: location.href });
    } catch (_) { /* user cancelled */ }
  } else if (navigator.clipboard) {
    await navigator.clipboard.writeText(text);
    showToast("Copied to clipboard");
  }
}

function toggleSpeak(report, isLeaf) {
  if (!("speechSynthesis" in window)) return;
  const btn = document.getElementById("btn-listen");
  if (state.speaking) {
    window.speechSynthesis.cancel();
    state.speaking = false;
    btn.innerHTML = `<span>🔊</span> ${t("btn_listen")}`;
    return;
  }
  const utter = new SpeechSynthesisUtterance(reportToText(report, isLeaf));
  const langMap = { en: "en-US", am: "am-ET", ti: "ti-ET", om: "om-ET" };
  utter.lang = langMap[state.lang] || "en-US";
  utter.onend = () => {
    state.speaking = false;
    btn.innerHTML = `<span>🔊</span> ${t("btn_listen")}`;
  };
  window.speechSynthesis.speak(utter);
  state.speaking = true;
  btn.innerHTML = `<span>⏹️</span> ${t("btn_stop_listen")}`;
}

/* ---------------------------------------------------------------------- */
/* History (localStorage)                                                  */
/* ---------------------------------------------------------------------- */

function loadHistory() {
  try {
    return JSON.parse(localStorage.getItem(STORAGE_KEYS.history) || "[]");
  } catch (_) {
    return [];
  }
}

function saveToHistory(data) {
  const list = loadHistory();
  const thumb = document.getElementById("preview-img").src;
  list.unshift({
    id: data.request_id,
    ts: Date.now(),
    class_key: data.class_key,
    name: data.report.name,
    severity: data.report.severity,
    lang: data.language,
    thumb: state.currentPreviewUrl && state.currentPreviewUrl.startsWith("blob:") ? null : thumb,
    data,
  });
  localStorage.setItem(STORAGE_KEYS.history, JSON.stringify(list.slice(0, MAX_HISTORY)));
}

function renderHistory() {
  const list = loadHistory();
  const container = document.getElementById("history-list");
  const emptyMsg = document.getElementById("history-empty");
  const clearBtn = document.getElementById("history-clear");

  if (list.length === 0) {
    container.innerHTML = "";
    emptyMsg.style.display = "block";
    clearBtn.style.display = "none";
    return;
  }
  emptyMsg.style.display = "none";
  clearBtn.style.display = "inline-flex";

  container.innerHTML = list
    .map((item, idx) => {
      const sev = item.severity || "none";
      const date = new Date(item.ts).toLocaleString(undefined, { dateStyle: "medium", timeStyle: "short" });
      return `
        <div class="history-item" data-idx="${idx}">
          ${item.thumb ? `<img src="${item.thumb}" alt="">` : `<div style="width:56px;height:56px;border-radius:10px;display:flex;align-items:center;justify-content:center;font-size:1.6rem;background:var(--color-surface-alt);">${CLASS_EMOJI[item.class_key] || "🌿"}</div>`}
          <div class="h-info">
            <strong>${escapeHtml(item.name)}</strong>
            <span>${date} · ${SEVERITY_EMOJI[sev] || ""} ${severityLabel(sev)}</span>
          </div>
        </div>`;
    })
    .join("");

  container.querySelectorAll(".history-item").forEach((el) => {
    el.addEventListener("click", () => {
      const idx = Number(el.dataset.idx);
      const item = list[idx];
      showView("scan");
      renderReport(item.data);
    });
  });
}

function clearHistory() {
  localStorage.removeItem(STORAGE_KEYS.history);
  renderHistory();
}

/* ---------------------------------------------------------------------- */
/* Library                                                                  */
/* ---------------------------------------------------------------------- */

async function renderLibrary() {
  const grid = document.getElementById("library-grid");
  grid.innerHTML = `<div class="skeleton" style="height:70px;margin-bottom:12px;"></div>`.repeat(4);
  try {
    let payload = state.libraryCache[state.lang];
    if (!payload) {
      const res = await fetch(`${API_BASE}/api/v1/diseases?lang=${state.lang}`);
      payload = await res.json();
      state.libraryCache[state.lang] = payload;
    }
    const entries = Object.entries(payload.classes);
    grid.innerHTML = entries
      .map(
        ([key, report]) => `
        <div class="library-card" data-key="${key}">
          <span class="emoji">${CLASS_EMOJI[key] || "🌿"}</span>
          <div>
            <strong>${escapeHtml(report.name)}</strong>
            <span>${escapeHtml((report.overview || "").slice(0, 90))}${report.overview && report.overview.length > 90 ? "…" : ""}</span>
          </div>
        </div>`
      )
      .join("");

    grid.querySelectorAll(".library-card").forEach((el) => {
      el.addEventListener("click", () => {
        const key = el.dataset.key;
        const report = payload.classes[key];
        showView("scan");
        renderReport({
          request_id: "library",
          language: state.lang,
          class_key: key,
          report,
          demo_mode: false,
          notice: null,
          detector: { is_coffee_leaf: true, confidence: 1, probabilities: {} },
          disease: { predicted_class: key, confidence: 1, probabilities: {} },
        });
        document.getElementById("upload-zone").style.display = "none";
        document.getElementById("tips-card").style.display = "none";
        document.getElementById("preview-wrap").style.display = "none";
      });
    });
  } catch (err) {
    grid.innerHTML = `<p>${t("error_network")}</p>`;
  }
}

/* ---------------------------------------------------------------------- */
/* Misc UI helpers                                                         */
/* ---------------------------------------------------------------------- */

let toastTimer = null;
function showToast(msg) {
  let toast = document.getElementById("toast");
  if (!toast) {
    toast = document.createElement("div");
    toast.id = "toast";
    toast.style.cssText =
      "position:fixed;left:50%;bottom:calc(var(--nav-h) + 16px);transform:translateX(-50%);" +
      "background:#2c2418;color:#fff;padding:10px 16px;border-radius:10px;font-size:0.85rem;" +
      "z-index:100;max-width:90%;text-align:center;box-shadow:0 4px 14px rgba(0,0,0,.25);";
    document.body.appendChild(toast);
  }
  toast.textContent = msg;
  toast.style.opacity = "1";
  clearTimeout(toastTimer);
  toastTimer = setTimeout(() => { toast.style.opacity = "0"; }, 3200);
}

function updateOnlineStatus() {
  document.getElementById("offline-banner").classList.toggle("show", !navigator.onLine);
}

/* ---------------------------------------------------------------------- */
/* Init                                                                     */
/* ---------------------------------------------------------------------- */

async function init() {
  initTheme();
  await setLanguage(state.lang);

  document.getElementById("lang-select").addEventListener("change", (e) => setLanguage(e.target.value));
  document.getElementById("theme-toggle").addEventListener("click", toggleTheme);

  document.querySelectorAll(".bottom-nav button").forEach((btn) => {
    btn.addEventListener("click", () => showView(btn.dataset.view));
  });

  // Upload
  const uploadZone = document.getElementById("upload-zone");
  const fileInput = document.getElementById("file-input");
  uploadZone.addEventListener("click", () => fileInput.click());
  document.getElementById("choose-file-btn").addEventListener("click", (e) => { e.stopPropagation(); fileInput.click(); });
  fileInput.addEventListener("change", (e) => handleFile(e.target.files[0]));

  ["dragenter", "dragover"].forEach((evt) =>
    uploadZone.addEventListener(evt, (e) => { e.preventDefault(); uploadZone.classList.add("dragover"); })
  );
  ["dragleave", "drop"].forEach((evt) =>
    uploadZone.addEventListener(evt, (e) => { e.preventDefault(); uploadZone.classList.remove("dragover"); })
  );
  uploadZone.addEventListener("drop", (e) => {
    const file = e.dataTransfer.files && e.dataTransfer.files[0];
    if (file) handleFile(file);
  });

  document.getElementById("retake-btn").addEventListener("click", resetUpload);
  document.getElementById("analyze-btn").addEventListener("click", analyze);

  // Camera
  document.getElementById("camera-btn").addEventListener("click", openCamera);
  document.getElementById("camera-capture-btn").addEventListener("click", captureFromCamera);
  document.getElementById("camera-switch-btn").addEventListener("click", switchCamera);
  document.getElementById("camera-close-btn").addEventListener("click", closeCamera);
  document.getElementById("camera-fallback-input").addEventListener("change", (e) => handleFile(e.target.files[0]));

  // History
  document.getElementById("history-clear").addEventListener("click", clearHistory);

  window.addEventListener("online", updateOnlineStatus);
  window.addEventListener("offline", updateOnlineStatus);
  updateOnlineStatus();

  if ("serviceWorker" in navigator) {
    navigator.serviceWorker.register("sw.js").catch(() => {});
  }

  showView("scan");
}

document.addEventListener("DOMContentLoaded", init);
