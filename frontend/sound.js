// sound.js - High-Volume UI Click (Capture Phase Enabled)

let audioCtx = null;

function getAudioContext() {
    if (!audioCtx) {
        audioCtx = new (window.AudioContext || window.webkitAudioContext)();
    }
    if (audioCtx.state === 'suspended') {
        audioCtx.resume();
    }
    return audioCtx;
}

function playClickSound() {
    const ctx = getAudioContext();
    const now = ctx.currentTime;

    const osc = ctx.createOscillator();
    const gain = ctx.createGain();

    // Loud, crisp tactile click
    osc.type = 'sine';
    osc.frequency.setValueAtTime(2200, now);
    osc.frequency.exponentialRampToValueAtTime(120, now + 0.03);

    gain.gain.setValueAtTime(0.9, now);
    gain.gain.exponentialRampToValueAtTime(0.001, now + 0.03);

    osc.connect(gain);
    gain.connect(ctx.destination);

    osc.start(now);
    osc.stop(now + 0.03);
}

// Captures ALL clicks BEFORE other scripts can block/stop event propagation
document.addEventListener('click', (event) => {
    const isClickable = event.target.closest(
        'button, a, select, option, label, input, li, .card, .clickable, ' +
        '.sidebar-item, .nav-item, .nav-link, [role="button"], [onclick], ' +
        '[class*="btn"], [class*="nav"], [class*="sidebar"], [class*="menu"], ' +
        '[class*="item"], [class*="link"], [class*="tab"], [class*="toggle"], ' +
        '[class*="inspect"], [id*="inspect"], canvas, svg, path'
    );

    if (isClickable) {
        playClickSound();
    }
}, true); // <--- 'true' = Intercepts clicks during CAPTURE phase