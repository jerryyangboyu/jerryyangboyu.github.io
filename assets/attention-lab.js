(() => {
  const root = document.querySelector('[data-attention-lab]');
  if (!root) return;

  const positionInput = root.querySelector('[data-query-position]');
  const positionReadout = root.querySelector('[data-position-readout]');
  const rows = root.querySelector('[data-attention-rows]');
  const output = root.querySelector('[data-attention-output]');
  const tokens = ['The', 'cat', 'sat', '.'];
  const scores = [0.7, 1.1, 0.2, 2.0];
  const values = [[1, 0], [0, 2], [1, 1], [-1, 1]];

  function softmax(xs) {
    const largest = Math.max(...xs);
    const exponentials = xs.map((x) => Math.exp(x - largest));
    const total = exponentials.reduce((sum, x) => sum + x, 0);
    return exponentials.map((x) => x / total);
  }

  function render() {
    const position = Number(positionInput.value);
    const allowedScores = scores.slice(0, position);
    const weights = softmax(allowedScores);
    const fullWeights = scores.map((_, index) => index < position ? weights[index] : 0);
    const blended = values[0].map((_, coordinate) =>
      fullWeights.reduce((sum, weight, index) => sum + weight * values[index][coordinate], 0)
    );

    positionReadout.textContent = String(position);
    rows.replaceChildren(...tokens.map((token, index) => {
      const tr = document.createElement('tr');
      const allowed = index < position;
      tr.innerHTML = '<td>' + (index + 1) + ': ' + token + '</td>' +
        '<td>' + scores[index].toFixed(1) + '</td>' +
        '<td class="' + (allowed ? '' : 'masked') + '">' + (allowed ? scores[index].toFixed(1) : '−∞') + '</td>' +
        '<td><strong>' + fullWeights[index].toFixed(3) + '</strong><br><progress max="1" value="' + fullWeights[index] + '"></progress></td>' +
        '<td>(' + values[index].join(', ') + ')</td>';
      return tr;
    }));
    output.textContent = '(' + blended.map((x) => x.toFixed(3)).join(', ') + ')';
  }

  positionInput.addEventListener('input', render);
  render();
})();
