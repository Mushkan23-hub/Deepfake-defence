// ============================================================
// Deepfake Defence — shared frontend behaviour
// ============================================================

(function () {
 try {
  // ---- Theme ----
  const root = document.documentElement;
  const saved = localStorage.getItem("dd-theme");
  if (saved) root.setAttribute("data-theme", saved);

  window.toggleTheme = function () {
    const current = root.getAttribute("data-theme") === "dark" ? "dark" : "light";
    const next = current === "dark" ? "light" : "dark";
    root.setAttribute("data-theme", next);
    localStorage.setItem("dd-theme", next);
    const btn = document.getElementById("theme-toggle-icon");
    if (btn) btn.textContent = next === "dark" ? "☀️" : "🌙";
  };

  document.addEventListener("DOMContentLoaded", () => {
    const btn = document.getElementById("theme-toggle-icon");
    if (btn) {
      const current = root.getAttribute("data-theme") === "dark" ? "dark" : "light";
      btn.textContent = current === "dark" ? "☀️" : "🌙";
    }
  });

  // ---- Sidebar ----
  window.toggleSidebar = function () {
    document.body.classList.toggle("sidebar-open");
  };
  window.closeSidebar = function () {
    document.body.classList.remove("sidebar-open");
  };
 } catch (e) { console.error("theme/sidebar init failed:", e); }
})();

// ============================================================
// Live cyber background — falling hex/binary "matrix rain" on
// the #cyber-bg canvas. Colour follows the current theme accent.
// ============================================================

(function () {
 try {
  const canvas = document.getElementById("cyber-bg");
  if (!canvas) { console.warn("cyber-bg canvas not found on this page"); return; }
  const ctx = canvas.getContext("2d");
  const reduceMotion = window.matchMedia("(prefers-reduced-motion: reduce)").matches;

  const CHARS = "01アイウエオカキクケコ01アイウ0123456789ABCDEF".split("");
  const FONT_SIZE = 15;
  let columns, drops, width, height;

  function resize() {
    width = canvas.width = window.innerWidth;
    height = canvas.height = window.innerHeight;
    columns = Math.floor(width / FONT_SIZE);
    drops = new Array(columns).fill(0).map(() => Math.floor(Math.random() * -40));
  }

  function accentColor() {
    return getComputedStyle(document.documentElement).getPropertyValue("--accent").trim() || "#4f5bff";
  }
  function bgColor() {
    return getComputedStyle(document.documentElement).getPropertyValue("--bg").trim() || "#f3f4f8";
  }

  function draw() {
    // Fading trail: paint a translucent rectangle of the theme background
    // over the previous frame instead of clearing it outright.
    ctx.fillStyle = hexToRgba(bgColor(), 0.14);
    ctx.fillRect(0, 0, width, height);

    ctx.font = FONT_SIZE + "px monospace";
    const color = accentColor();

    for (let i = 0; i < columns; i++) {
      const char = CHARS[Math.floor(Math.random() * CHARS.length)];
      const x = i * FONT_SIZE;
      const y = drops[i] * FONT_SIZE;

      // Leading character brighter, rest of trail fainter (handled by the fade fill above).
      ctx.fillStyle = hexToRgba(color, y < FONT_SIZE * 2 ? 0.9 : 0.5);
      ctx.fillText(char, x, y);

      if (y > height && Math.random() > 0.975) {
        drops[i] = 0;
      }
      drops[i]++;
    }
  }

  function hexToRgba(hex, alpha) {
    hex = hex.replace("#", "");
    if (hex.length === 3) hex = hex.split("").map(c => c + c).join("");
    const num = parseInt(hex, 16);
    if (isNaN(num)) return `rgba(79,91,255,${alpha})`;
    const r = (num >> 16) & 255, g = (num >> 8) & 255, b = num & 255;
    return `rgba(${r},${g},${b},${alpha})`;
  }

  resize();
  window.addEventListener("resize", resize);

  if (reduceMotion) {
    // Static single faint frame instead of a running animation.
    ctx.fillStyle = bgColor();
    ctx.fillRect(0, 0, width, height);
  } else {
    setInterval(draw, 55);
  }
 } catch (e) { console.error("cyber-bg animation failed:", e); }
})();

// ============================================================
// Typewriter effect — cycles through a list of phrases into any
// element with id="typewriter-target".
// ============================================================

(function () {
 try {
  const el = document.getElementById("typewriter-target");
  if (!el) { console.warn("typewriter-target not found on this page"); return; }

  const phrases = [
    "Detecting deepfakes with AI-driven image, video and audio analysis.",
    "Recovering hidden data with Kali Linux steganography and file carving.",
    "Investigating networks with PCAP analysis and gated port scanning.",
    "Running OSINT lookups tied to an authorised, auditable case file.",
  ];

  let phraseIndex = 0, charIndex = 0, deleting = false;

  function tick() {
    const phrase = phrases[phraseIndex];
    if (!deleting) {
      charIndex++;
      el.textContent = phrase.slice(0, charIndex);
      if (charIndex === phrase.length) {
        deleting = true;
        setTimeout(tick, 1600);
        return;
      }
    } else {
      charIndex--;
      el.textContent = phrase.slice(0, charIndex);
      if (charIndex === 0) {
        deleting = false;
        phraseIndex = (phraseIndex + 1) % phrases.length;
      }
    }
    setTimeout(tick, deleting ? 22 : 38);
  }
  tick();
 } catch (e) { console.error("typewriter init failed:", e); }
})();

// ---- Generic helpers for tool pages ----

function ddFormatBytes(n) {
  if (n === undefined || n === null) return "";
  if (n < 1024) return n + " B";
  if (n < 1024 * 1024) return (n / 1024).toFixed(1) + " KB";
  return (n / (1024 * 1024)).toFixed(2) + " MB";
}

function ddSetLoading(btn, loading, label) {
  if (!btn) return;
  btn.disabled = loading;
  btn.innerHTML = loading
    ? '<span class="spinner"></span> Running…'
    : (label || btn.dataset.label || "Run");
}

/**
 * Runs a Kali file-based tool: uploads the selected file (or a stored_name),
 * shows a spinner, renders the JSON result.
 */
async function ddRunFileTool(tool, opts) {
  const { fileInput, caseSelect, button, resultEl, hashEl, artifactsEl } = opts;
  const file = fileInput && fileInput.files[0];
  if (!file) {
    resultEl.textContent = "Choose a file first.";
    return;
  }
  const fd = new FormData();
  fd.append("file", file);
  if (caseSelect && caseSelect.value) fd.append("case_id", caseSelect.value);

  ddSetLoading(button, true);
  resultEl.textContent = "";
  try {
    const res = await fetch(`/api/kali/file/${tool}`, { method: "POST", body: fd });
    const data = await res.json();
    renderToolResult(data, resultEl, hashEl, artifactsEl);
  } catch (err) {
    resultEl.textContent = "Request failed: " + err;
  } finally {
    ddSetLoading(button, false);
  }
}

async function ddRunJsonTool(url, payload, opts) {
  const { button, resultEl } = opts;
  ddSetLoading(button, true);
  resultEl.textContent = "";
  try {
    const res = await fetch(url, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
    });
    const data = await res.json();
    renderToolResult(data, resultEl);
  } catch (err) {
    resultEl.textContent = "Request failed: " + err;
  } finally {
    ddSetLoading(button, false);
  }
}

function renderToolResult(data, resultEl, hashEl, artifactsEl) {
  if (data.error) {
    resultEl.innerHTML = `<span style="color:var(--danger)">${escapeHtml(data.error)}</span>` +
      (data.hint ? `<br><span style="color:var(--text-muted)">${escapeHtml(data.hint)}</span>` : "");
    return;
  }

  let out = "";
  if (data.command) out += "$ " + data.command + "\n\n";
  out += (data.stdout || "(no output)").trim();
  if (data.stderr) out += "\n\n--- stderr ---\n" + data.stderr.trim();
  resultEl.textContent = out;

  if (hashEl && data.hashes) {
    hashEl.innerHTML = Object.entries(data.hashes)
      .map(([k, v]) => `<div><strong>${k}:</strong> ${v}</div>`)
      .join("");
  }
  if (artifactsEl) {
    if (data.artifacts && data.artifacts.length) {
      artifactsEl.innerHTML = "<table><tr><th>File</th><th>Size</th></tr>" +
        data.artifacts.map(a => `<tr><td>${escapeHtml(a.name)}</td><td>${ddFormatBytes(a.size)}</td></tr>`).join("") +
        "</table>";
    } else {
      artifactsEl.innerHTML = "";
    }
  }
}

function escapeHtml(str) {
  const div = document.createElement("div");
  div.textContent = str;
  return div.innerHTML;
}
