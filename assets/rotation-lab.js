(() => {
  const colors = { ink: '#17232b', muted: '#7a878d', teal: '#087e78', coral: '#c54f3f', grid: '#dce2e1' };

  function drawArrow(ctx, x0, y0, x1, y1, color, label) {
    const angle = Math.atan2(y1 - y0, x1 - x0);
    ctx.strokeStyle = color;
    ctx.fillStyle = color;
    ctx.lineWidth = 3;
    ctx.beginPath();
    ctx.moveTo(x0, y0);
    ctx.lineTo(x1, y1);
    ctx.stroke();
    ctx.beginPath();
    ctx.moveTo(x1, y1);
    ctx.lineTo(x1 - 12 * Math.cos(angle - Math.PI / 6), y1 - 12 * Math.sin(angle - Math.PI / 6));
    ctx.lineTo(x1 - 12 * Math.cos(angle + Math.PI / 6), y1 - 12 * Math.sin(angle + Math.PI / 6));
    ctx.closePath();
    ctx.fill();
    ctx.font = '700 14px system-ui';
    ctx.fillText(label, x1 + 8 * Math.cos(angle), y1 + 8 * Math.sin(angle) - 8);
  }

  document.querySelectorAll('[data-rotation-lab]').forEach((root) => {
    const canvas = root.querySelector('canvas');
    const slider = root.querySelector('input[type="range"]');
    const readout = root.querySelector('[data-angle-readout]');
    const x = Number(root.dataset.x || 1);
    const y = Number(root.dataset.y || 0);

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
      const scale = Math.min(width, height) * 0.3 / Math.max(1, Math.hypot(x, y));
      const degrees = Number(slider.value);
      const theta = degrees * Math.PI / 180;
      const xr = x * Math.cos(theta) - y * Math.sin(theta);
      const yr = x * Math.sin(theta) + y * Math.cos(theta);

      ctx.clearRect(0, 0, width, height);
      ctx.strokeStyle = colors.grid;
      ctx.lineWidth = 1;
      for (let r = 1; r <= 3; r += 1) {
        ctx.beginPath();
        ctx.arc(cx, cy, r * scale / 3, 0, Math.PI * 2);
        ctx.stroke();
      }
      ctx.strokeStyle = colors.muted;
      ctx.beginPath();
      ctx.moveTo(24, cy); ctx.lineTo(width - 24, cy);
      ctx.moveTo(cx, 18); ctx.lineTo(cx, height - 18);
      ctx.stroke();
      ctx.fillStyle = colors.muted;
      ctx.font = '13px system-ui';
      ctx.fillText('x', width - 32, cy - 8);
      ctx.fillText('y', cx + 8, 30);

      drawArrow(ctx, cx, cy, cx + x * scale, cy - y * scale, colors.ink, 'v');
      drawArrow(ctx, cx, cy, cx + xr * scale, cy - yr * scale, colors.coral, 'Rv');

      readout.textContent = `${degrees}° · ${(theta).toFixed(2)} rad`;
      const output = root.querySelector('[data-vector-readout]');
      if (output) output.textContent = `(${xr.toFixed(2)}, ${yr.toFixed(2)})`;
    }

    slider.addEventListener('input', draw);
    window.addEventListener('resize', draw);
    draw();
  });
})();
