const { ipcRenderer } = require('electron');
const nodePath = require('path');

const bgPath    = nodePath.join(__dirname, '..', '..', '..', 'asset', 'background.png').replace(/\\/g, '/');
const assetBase = nodePath.join(__dirname, '..', '..', '..', 'asset').replace(/\\/g, '/');

document.body.style.backgroundImage    = `url("${bgPath}")`;
document.body.style.backgroundSize     = 'cover';
document.body.style.backgroundPosition = 'center';
document.body.style.backgroundRepeat   = 'no-repeat';

let videoLoaded       = false;
let videoElements     = [];
let isPlaying         = false;
let activeZoomElement = null;
let zoomPlaceholder   = null;

const SVG_PLAY   = `<svg viewBox="0 0 24 24"><polygon points="5 3 19 12 5 21 5 3"/></svg>`;
const SVG_PAUSE  = `<svg viewBox="0 0 24 24"><rect x="6" y="4" width="4" height="16"/><rect x="14" y="4" width="4" height="16"/></svg>`;
const SVG_REWIND = `<svg viewBox="0 0 24 24"><polygon points="11 19 2 12 11 5 11 19"/><polygon points="22 19 13 12 22 5 22 19"/></svg>`;

function safeGet(id) {
    return document.getElementById(id);
}

function checkAnalyzeRequirements() {
    const btnAnalyze = document.getElementById('btnAnalyze');
    if (!btnAnalyze) return;

    let canAnalyze = videoLoaded && videoElements.length > 0;
    const speedInputs = document.querySelectorAll('.sidebar-speed-input');
    if (speedInputs.length > 0) {
        speedInputs.forEach(input => {
            if (input.value.trim() === "" || parseFloat(input.value) <= 0) canAnalyze = false;
        });
    }

    if (canAnalyze) {
        btnAnalyze.disabled = false;
        btnAnalyze.classList.add('active-cta');
        btnAnalyze.style.cursor  = "pointer";
        btnAnalyze.style.opacity = "1";
    } else {
        btnAnalyze.disabled = true;
        btnAnalyze.classList.remove('active-cta');
        btnAnalyze.style.cursor  = "not-allowed";
        btnAnalyze.style.opacity = "0.6";
    }
}

async function importVideos() {
    const paths = await ipcRenderer.invoke('open-file-dialog');
    if (!paths || paths.length === 0) return;

    const grid           = safeGet('gridContainer');
    const globalSeek     = safeGet('globalSeek');
    const speedPanel     = safeGet('speedPanel');
    const speedContainer = safeGet('speedRowsContainer');

    if (grid)           grid.innerHTML = '';
    if (globalSeek)     globalSeek.value = 0;
    if (speedContainer) speedContainer.innerHTML = '';
    if (speedPanel)     speedPanel.style.display = 'block';

    videoElements = [];
    videoLoaded   = true;

    const btnRemove = safeGet('btnRemove');
    if (btnRemove) btnRemove.disabled = false;

    paths.slice(0, 4).forEach((filePath, index) => {
        const fileUrl     = `file://${filePath}`;
        const wrapper     = document.createElement('div');
        wrapper.className = 'video-wrapper';
        wrapper.id        = `wrapper-${index}`;
        wrapper.innerHTML = `
            <div class="cam-badge">CAM ${index + 1}</div>
            <video src="${fileUrl}" id="vid-${index}" onclick="toggleZoom('wrapper-${index}')" muted loop playsinline></video>
            <div class="mini-controls">
                <button class="mini-btn" onclick="rewindSingle(${index}, event)" title="Riavvolgi">${SVG_REWIND}</button>
                <button class="mini-btn" id="btn-play-${index}" onclick="playSingle(${index}, event)" title="Play/Pause">${SVG_PLAY}</button>
                <input type="range" id="seek-${index}" min="0" max="100" value="0"
                    oninput="seekSingle(${index}, this.value, event)"
                    onclick="event.stopPropagation()">
            </div>`;
        if (grid) grid.appendChild(wrapper);

        if (speedContainer) {
            const row     = document.createElement('div');
            row.className = 'speed-row';
            row.innerHTML = `
                <span class="cam-label-text">CAM ${index + 1}</span>
                <input type="number" class="sidebar-speed-input" id="speed-input-${index}"
                    step="0.01" min="0.1" value="1.00" oninput="checkAnalyzeRequirements()">`;
            speedContainer.appendChild(row);
        }

        setTimeout(() => {
            const video = document.getElementById(`vid-${index}`);
            if (video) {
                videoElements.push(video);
                video.addEventListener('timeupdate', () => {
                    const seek = document.getElementById(`seek-${index}`);
                    if (seek && video.duration) seek.value = (video.currentTime / video.duration) * 100;
                    if (index === 0) {
                        const gSeek = document.getElementById('globalSeek');
                        if (gSeek && video.duration) gSeek.value = (video.currentTime / video.duration) * 100;
                    }
                });
            }
        }, 50);
    });

    const layoutClass = paths.length === 1 ? 'one-video'
                      : paths.length === 2 ? 'two-videos'
                      : paths.length === 3 ? 'three-videos' : '';
    if (grid)           grid.className = `video-grid ${layoutClass}`;
    if (speedContainer) speedContainer.className = `speed-rows-container ${layoutClass}`;

    const globalControls = safeGet('globalControls');
    if (globalControls) {
        globalControls.style.opacity       = '1';
        globalControls.style.pointerEvents = 'auto';
    }

    isPlaying = true;
    setTimeout(() => {
        checkAnalyzeRequirements();
        videoElements.forEach(v => { if (v) v.play().catch(() => {}); });
        updateGlobalIcons();
        videoElements.forEach((_, idx) => updateSingleIcon(idx));
    }, 200);
}

function clearVideos() {
    if (activeZoomElement) {
        if (activeZoomElement.parentNode) activeZoomElement.parentNode.removeChild(activeZoomElement);
        activeZoomElement = null;
        zoomPlaceholder   = null;
    }

    const container = safeGet('gridContainer');
    if (container) {
        container.className = 'video-grid empty';
        container.classList.remove('has-zoom');
        container.innerHTML = `<div class="empty-state-premium"><p>Upload up to 4 videos to start the session.</p></div>`;
    }

    videoElements = [];
    videoLoaded   = false;
    checkAnalyzeRequirements();

    const gSeek = document.getElementById('globalSeek');
    if (gSeek) gSeek.value = 0;

    const btnRemove = document.getElementById('btnRemove');
    if (btnRemove) btnRemove.disabled = true;

    const spdPanel = document.getElementById('speedPanel');
    if (spdPanel) spdPanel.style.display = 'none';

    const spdCont = document.getElementById('speedRowsContainer');
    if (spdCont) { spdCont.innerHTML = ''; spdCont.className = 'speed-rows-container'; }

    const gControls = document.getElementById('globalControls');
    if (gControls) { gControls.style.opacity = '0'; gControls.style.pointerEvents = 'none'; }

    const aiPanel = document.getElementById('aiPanel');
    if (aiPanel) aiPanel.classList.remove('active');

    const vSev = document.getElementById('valSeverity');
    if (vSev) { vSev.innerText = '-'; vSev.style.color = 'white'; }

    const vOff = document.getElementById('valOffence');
    if (vOff) { vOff.innerText = '-'; vOff.style.color = 'white'; }

    const vAct = document.getElementById('valAction');
    if (vAct) vAct.innerText = '-';

    const overlay = document.getElementById('analysisOverlay');
    if (overlay) { overlay.classList.remove('active'); overlay.style.display = 'none'; }

    isPlaying = false;
    updateGlobalIcons();
}

function toggleZoom(wrapperId) {
    const el = document.getElementById(wrapperId);
    if (!el || el.classList.contains('is-animating')) return;
    if (el.classList.contains('zoomed')) {
        zoomOut(el);
    } else {
        if (activeZoomElement && activeZoomElement !== el) zoomOut(activeZoomElement, true);
        zoomIn(el);
    }
}

function zoomIn(el) {
    const container = document.querySelector('.stage-area');
    const grid      = document.getElementById('gridContainer');
    if (grid) grid.classList.add('has-zoom');

    const startRect     = el.getBoundingClientRect();
    const containerRect = container.getBoundingClientRect();

    zoomPlaceholder           = document.createElement('div');
    zoomPlaceholder.className = 'video-wrapper placeholder';
    el.parentNode.insertBefore(zoomPlaceholder, el);

    el.classList.add('is-animating');
    el.style.position = 'absolute';
    el.style.top      = (startRect.top  - containerRect.top)  + 'px';
    el.style.left     = (startRect.left - containerRect.left) + 'px';
    el.style.width    = startRect.width  + 'px';
    el.style.height   = startRect.height + 'px';
    el.style.margin   = '0';
    container.appendChild(el);

    el.offsetHeight;
    el.classList.add('zoomed');
    el.style.top    = '10px';
    el.style.left   = '10px';
    el.style.width  = (containerRect.width  - 20) + 'px';
    el.style.height = (containerRect.height - 20) + 'px';

    activeZoomElement = el;
    setTimeout(() => el.classList.remove('is-animating'), 550);
}

function zoomOut(el, instant = false) {
    if (!zoomPlaceholder) { fallbackZoomReset(el); return; }

    const container     = document.querySelector('.stage-area');
    const grid          = document.getElementById('gridContainer');
    const containerRect = container.getBoundingClientRect();
    const targetRect    = zoomPlaceholder.getBoundingClientRect();

    if (grid) grid.classList.remove('has-zoom');
    if (instant) { resetZoomState(el); return; }

    el.classList.add('is-animating');
    el.classList.remove('zoomed');
    el.style.top    = (targetRect.top  - containerRect.top)  + 'px';
    el.style.left   = (targetRect.left - containerRect.left) + 'px';
    el.style.width  = targetRect.width  + 'px';
    el.style.height = targetRect.height + 'px';

    setTimeout(() => resetZoomState(el), 550);
}

function resetZoomState(el) {
    el.classList.remove('is-animating');
    el.classList.remove('zoomed');
    el.style = '';
    if (zoomPlaceholder) {
        if (zoomPlaceholder.parentNode) zoomPlaceholder.parentNode.replaceChild(el, zoomPlaceholder);
    } else {
        const grid = document.getElementById('gridContainer');
        if (grid) grid.appendChild(el);
    }
    zoomPlaceholder   = null;
    activeZoomElement = null;
}

function fallbackZoomReset(el) {
    el.classList.remove('zoomed');
    el.classList.remove('is-animating');
    el.style = '';
    const grid = document.getElementById('gridContainer');
    if (grid) { grid.appendChild(el); grid.classList.remove('has-zoom'); }
    activeZoomElement = null;
}

function togglePlayAll() {
    isPlaying = !isPlaying;
    videoElements.forEach(v => { if (v) isPlaying ? v.play() : v.pause(); });
    updateGlobalIcons();
    videoElements.forEach((_, idx) => updateSingleIcon(idx));
}

function rewindAll() {
    videoElements.forEach(v => { if (v) v.currentTime = 0; });
}

function seekAll(val) {
    videoElements.forEach(v => { if (v && v.duration) v.currentTime = (val / 100) * v.duration; });
}

function updateGlobalIcons() {
    const iconPlay  = document.getElementById('iconPlay');
    const iconPause = document.getElementById('iconPause');
    if (iconPlay && iconPause) {
        iconPlay.style.display  = isPlaying ? 'none'  : 'block';
        iconPause.style.display = isPlaying ? 'block' : 'none';
    }
}

function playSingle(idx, event) {
    event.stopPropagation();
    const v = videoElements[idx];
    if (v) { if (v.paused) v.play(); else v.pause(); }
    updateSingleIcon(idx);
}

function rewindSingle(idx, event) {
    event.stopPropagation();
    const v = videoElements[idx];
    if (v) v.currentTime = 0;
}

function seekSingle(idx, val, event) {
    event.stopPropagation();
    const v = videoElements[idx];
    if (v && v.duration) v.currentTime = (val / 100) * v.duration;
}

function updateSingleIcon(idx) {
    const btn = document.getElementById(`btn-play-${idx}`);
    const v   = videoElements[idx];
    if (btn && v) btn.innerHTML = v.paused ? SVG_PLAY : SVG_PAUSE;
}

async function runAnalysis() {
    if (!videoElements || videoElements.length === 0) { alert('Video not imported!'); return; }

    const aiPanel    = safeGet('aiPanel');
    const elSeverity = safeGet('valSeverity');
    const elOffence  = safeGet('valOffence');
    const elAction   = safeGet('valAction');
    const overlay    = safeGet('analysisOverlay');

    if (aiPanel)    aiPanel.classList.add('active');
    if (elSeverity) { elSeverity.innerText = 'ENSEMBLE...'; elSeverity.style.color = 'white'; }
    if (elOffence)  { elOffence.innerText = 'WAIT...'; elOffence.style.color = '#FFFFFF'; }
    if (elAction)   elAction.innerText = 'WAIT...';
    if (overlay)    { overlay.style.display = 'none'; overlay.classList.remove('active'); }

    try {
        let allVideoPaths  = [];
        let allVideoSpeeds = [];

        for (let i = 0; i < videoElements.length; i++) {
            let rawPath = videoElements[i].getAttribute('src');
            if (rawPath) {
                if (rawPath.startsWith('file://')) rawPath = rawPath.replace('file://', '');
                rawPath = decodeURIComponent(rawPath);
                allVideoPaths.push(rawPath);
            }
            const speedInput = document.getElementById(`speed-input-${i}`);
            let speedValue   = speedInput ? parseFloat(speedInput.value) : 1.0;
            if (isNaN(speedValue) || speedValue <= 0) speedValue = 1.0;
            allVideoSpeeds.push(speedValue);
        }

        if (allVideoPaths.length === 0) throw new Error('Path not found');

        const response = await fetch('http://127.0.0.1:5000/analyze', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ video_paths: allVideoPaths, speeds: allVideoSpeeds }),
        });

        if (!response.ok) throw new Error(`Status: ${response.status}`);
        const data = await response.json();
        if (data.error) throw new Error(data.error);

        const assets = [ '../../asset/no_card.png', '../../asset/yellow_card.png', '../../asset/red_card.png'];
        const colors         = ['#FFFFFF', '#FFD60A', '#FF453A'];
        const severityLabels = ['NO CARD', 'YELLOW CARD', 'RED CARD'];

        const offenceText  = data.is_foul ? 'OFFENCE'  : 'NO OFFENCE';
        const offenceColor = data.is_foul ? '#FF453A'  : '#4CD964';
        const severityText = severityLabels[data.severity];
        const actionLabel  = data.action_class || 'UNKNOWN';

        const sevPerc = (data.severity_conf * 100).toFixed(0);
        const offPerc = (data.offence_conf  * 100).toFixed(0);
        const actPerc = (data.action_conf   * 100).toFixed(0);

        if (elSeverity) { elSeverity.innerText = `${severityText} ${sevPerc}%`; elSeverity.style.color = colors[data.severity]; }
        if (elOffence)  { elOffence.innerText  = `${offenceText} ${offPerc}%`;  elOffence.style.color  = offenceColor; }
        if (elAction)   elAction.innerText = `${actionLabel} ${actPerc}%`;

        const img     = safeGet('resultImage');
        const resText = safeGet('resultText');
        if (img)     { img.src = assets[data.severity] || assets[0]; img.style.display = 'block'; }
        if (resText) { resText.innerText = severityText; resText.style.color = colors[data.severity]; }

        if (overlay) {
            overlay.style.display = 'flex';
            setTimeout(() => overlay.classList.add('active'), 10);
        }

    } catch (error) {
        if (elOffence) { elOffence.innerText = 'ERRORE'; elOffence.style.color = '#FF453A'; }
        if (elAction)  elAction.innerText = 'Controlla Console';
        alert(`Errore Analisi: ${error.message}`);
    }
}

function closeAnalysis() {
    const overlay = document.getElementById('analysisOverlay');
    if (overlay) {
        overlay.classList.remove('active');
        setTimeout(() => { overlay.style.display = 'none'; }, 600);
    }
}