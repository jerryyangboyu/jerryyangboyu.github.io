(() => {
  const lab = document.querySelector('[data-temperature-lab]');
  if (!lab) return;

  const tokens = ['the', 'cat', 'sat', 'on', 'mat'];
  const logits = [2.1, 1.3, 0.2, -0.4, -1.2];
  const slider = lab.querySelector('[data-temperature]');
  const value = lab.querySelector('[data-temperature-value]');
  const bars = lab.querySelector('[data-sampling-bars]');

  const render = () => {
    const temperature = Number(slider.value);
    const scaled = logits.map((logit) => logit / temperature);
    const largest = Math.max(...scaled);
    const exponentials = scaled.map((logit) => Math.exp(logit - largest));
    const total = exponentials.reduce((sum, item) => sum + item, 0);
    const probabilities = exponentials.map((item) => item / total);

    value.textContent = temperature.toFixed(1);
    bars.innerHTML = probabilities.map((probability, index) =>
      `<div class="sampling-row"><strong>${tokens[index]}</strong><span class="sampling-track"><span class="sampling-fill" style="--probability:${(probability * 100).toFixed(1)}%"></span></span><span>${(probability * 100).toFixed(1)}%</span></div>`
    ).join('');
  };

  slider.addEventListener('input', render);
  render();
})();
