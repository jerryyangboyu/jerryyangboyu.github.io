(() => {
  const root = document.querySelector('[data-rope-lab]');
  if (!root) return;

  const canvas = root.querySelector('canvas');
  const mInput = root.querySelector('[data-m]');
  const nInput = root.querySelector('[data-n]');
  const thetaInput = root.querySelector('[data-theta]');
  const readout = root.querySelector('[data-rope-readout]');
  const colors = { ink: '#17232b', muted: '#7a878d', blue: '#315ea8', coral: '#c54f3f', teal: '#087e78', grid: '#dce2e1' };

  function arrow(ctx, cx, cy, radius, angle, color, label) {
    const x = cx + radius * Math.cos(angle);
    const y = cy - radius * Math.sin(angle);
    ctx.strokeStyle = color;
    ctx.fillStyle = color;
    ctx.lineWidth = 3;
    ctx.beginPath(); ctx.moveTo(cx, cy); ctx.lineTo(x, y); ctx.stroke();
    const a = Math.atan2(y - cy, x - cx);
    ctx.beginPath();
    ctx.moveTo(x, y);
    ctx.lineTo(x - 11 * Math.cos(a - Math.PI / 6), y - 11 * Math.sin(a - Math.PI / 6));
    ctx.lineTo(x - 11 * Math.cos(a + Math.PI / 6), y - 11 * Math.sin(a + Math.PI / 6));
    ctx.closePath(); ctx.fill();
    ctx.font = '700 13px system-ui';
    ctx.fillText(label, x + 8, y - 8);
  }

  function draw() {
    const rect = canvas.getBoundingClientRect();
    const ratio = window.devicePixelRatio || 1;
    canvas.width = Math.max(1, Math.round(rect.width * ratio));
    canvas.height = Math.max(1, Math.round(rect.height * ratio));
    const ctx = canvas.getContext('2d');
    ctx.scale(ratio, ratio);
    const width = rect.width;
    const height = rect.height;
    const cx = width / 2;
    const cy = height / 2;
    const radius = Math.min(width, height) * 0.31;
    const m = Number(mInput.value);
    const n = Number(nInput.value);
    const theta = Number(thetaInput.value);
    const am = m * theta;
    const an = n * theta;
    const relative = (n - m) * theta;

    ctx.clearRect(0, 0, width, height);
    ctx.strokeStyle = colors.grid;
    ctx.lineWidth = 1;
    ctx.beginPath(); ctx.arc(cx, cy, radius, 0, Math.PI * 2); ctx.stroke();
    ctx.beginPath(); ctx.moveTo(cx - radius - 18, cy); ctx.lineTo(cx + radius + 18, cy); ctx.stroke();
    ctx.beginPath(); ctx.moveTo(cx, cy - radius - 18); ctx.lineTo(cx, cy + radius + 18); ctx.stroke();

    arrow(ctx, cx, cy, radius, am, colors.blue, 'Rₘq');
    arrow(ctx, cx, cy, radius * 0.86, an, colors.coral, 'Rₙk');
    arrow(ctx, cx, cy, radius * 0.68, relative, colors.teal, 'Rₙ₋ₘ');

    readout.innerHTML = `<span><strong>mθ</strong> = ${am.toFixed(2)}</span><span><strong>nθ</strong> = ${an.toFixed(2)}</span><span>relative angle <strong>(n−m)θ = ${relative.toFixed(2)} rad</strong></span>`;
  }

  [mInput, nInput, thetaInput].forEach((input) => input.addEventListener('input', draw));
  window.addEventListener('resize', draw);
  draw();
})();
