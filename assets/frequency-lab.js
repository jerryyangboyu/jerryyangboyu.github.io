(() => {
  const root = document.querySelector('[data-frequency-lab]');
  if (!root) return;

  const canvas = root.querySelector('canvas');
  const positionInput = root.querySelector('input[type="range"]');
  const readout = root.querySelector('[data-position-readout]');
  const frequencies = [1, 0.1, 0.01, 0.001];
  const colors = ['#c54f3f', '#315ea8', '#087e78', '#a96c00'];

  function draw() {
    const rect = canvas.getBoundingClientRect();
    const ratio = window.devicePixelRatio || 1;
    canvas.width = Math.max(1, Math.round(rect.width * ratio));
    canvas.height = Math.max(1, Math.round(rect.height * ratio));
    const ctx = canvas.getContext('2d');
    ctx.scale(ratio, ratio);
    const width = rect.width;
    const height = rect.height;
    const m = Number(positionInput.value);
    const lane = height / frequencies.length;
    const radius = Math.min(34, lane * 0.32);

    ctx.clearRect(0, 0, width, height);
    frequencies.forEach((theta, i) => {
      const cy = lane * (i + 0.5);
      const cx = Math.min(width * 0.68, width - radius - 86);
      const angle = m * theta;
      ctx.strokeStyle = '#dce2e1';
      ctx.lineWidth = 1;
      ctx.beginPath();
      ctx.moveTo(120, cy);
      ctx.lineTo(cx - radius - 20, cy);
      ctx.stroke();
      ctx.beginPath();
      ctx.arc(cx, cy, radius, 0, Math.PI * 2);
      ctx.stroke();
      ctx.strokeStyle = colors[i];
      ctx.fillStyle = colors[i];
      ctx.lineWidth = 3;
      ctx.beginPath();
      ctx.moveTo(cx, cy);
      ctx.lineTo(cx + radius * Math.cos(angle), cy - radius * Math.sin(angle));
      ctx.stroke();
      ctx.beginPath();
      ctx.arc(cx + radius * Math.cos(angle), cy - radius * Math.sin(angle), 4, 0, Math.PI * 2);
      ctx.fill();
      ctx.fillStyle = '#17232b';
      ctx.font = '700 13px system-ui';
      ctx.fillText('pair ' + i, 18, cy - 4);
      ctx.fillStyle = '#5a6870';
      ctx.font = '12px ui-monospace, monospace';
      ctx.fillText('theta=' + theta, 18, cy + 15);
      ctx.fillText(angle.toFixed(2) + ' rad', cx + radius + 14, cy + 4);
    });
    readout.textContent = 'position m = ' + m;
  }

  positionInput.addEventListener('input', draw);
  window.addEventListener('resize', draw);
  draw();
})();
