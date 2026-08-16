(() => {
  const format = (value) => {
    const clean = Math.abs(value) < 0.0005 ? 0 : value;
    return Number(clean.toFixed(3)).toString();
  };

  const setPressed = (buttons, active) => {
    buttons.forEach((button) => button.setAttribute('aria-pressed', String(button === active)));
  };

  const cells = (values, className = 'feature-cell') =>
    values.map((value) => '<span class="' + className + '">' + format(value) + '</span>').join('');

  document.querySelectorAll('[data-tensor-viz]').forEach((root) => {
    const batches = [
      [[0.4, -0.2, 1.1, 0, 0.7, -0.5], [0.1, 0.8, -0.3, 1.2, 0.2, 0.4], [-0.6, 0.3, 0.9, -0.1, 1.4, 0.2], [0.2, -0.7, 0.5, 0.8, -0.4, 1]],
      [[-0.3, 0.6, 0.1, 1.3, -0.2, 0.8], [0.7, 0.2, -0.5, 0.4, 1.1, -0.1], [1, -0.4, 0.3, 0.6, 0.2, -0.8], [-0.2, 1.2, 0.4, -0.6, 0.9, 0.1]]
    ];
    const labels = ['The', 'cat', 'sat', '.'];
    const grid = root.querySelector('[data-tensor-grid]');
    const readout = root.querySelector('[data-tensor-readout]');
    const batchButtons = [...root.querySelectorAll('[data-batch]')];
    let batch = 0;
    let token = 1;

    const render = () => {
      grid.innerHTML = batches[batch].map((row, index) =>
        '<button type="button" class="tensor-row" data-token="' + index + '" aria-pressed="' + (index === token) + '">' +
        '<span class="token-label">' + (index + 1) + ' ' + labels[index] + '</span>' + cells(row, 'tensor-cell') + '</button>'
      ).join('');
      readout.textContent = 'X[' + (batch + 1) + ', ' + (token + 1) + ', :] selects token “' + labels[token] + '”: (' + batches[batch][token].map(format).join(', ') + ')';
      grid.querySelectorAll('[data-token]').forEach((button) => button.addEventListener('click', () => {
        token = Number(button.dataset.token);
        render();
      }));
    };

    batchButtons.forEach((button) => button.addEventListener('click', () => {
      batch = Number(button.dataset.batch);
      setPressed(batchButtons, button);
      render();
    }));
    render();
  });

  document.querySelectorAll('[data-norm-viz]').forEach((root) => {
    const input = [1, 2, 3];
    const modes = [...root.querySelectorAll('[data-norm-mode]')];
    const before = root.querySelector('[data-norm-before]');
    const after = root.querySelector('[data-norm-after]');
    const readout = root.querySelector('[data-norm-readout]');

    const bars = (values) => values.map((value, index) => {
      const magnitude = Math.min(1, Math.abs(value) / 3);
      return '<div class="value-bar"><span>f' + (index + 1) + '</span><span class="bar-track"><span class="bar-fill ' +
        (value >= 0 ? 'positive' : 'negative') + '" style="--magnitude:' + magnitude + '"></span></span><strong>' + format(value) + '</strong></div>';
    }).join('');

    const render = (mode) => {
      const mean = input.reduce((sum, x) => sum + x, 0) / input.length;
      const variance = input.reduce((sum, x) => sum + (x - mean) ** 2, 0) / input.length;
      const rms = Math.sqrt(input.reduce((sum, x) => sum + x ** 2, 0) / input.length);
      const output = mode === 'layer'
        ? input.map((x) => (x - mean) / Math.sqrt(variance))
        : input.map((x) => x / rms);
      before.innerHTML = bars(input);
      after.innerHTML = bars(output);
      readout.textContent = mode === 'layer'
        ? 'LayerNorm subtracts μ = 2, then divides by σ ≈ 0.816.'
        : 'RMSNorm keeps the offset and divides by RMS ≈ 2.160.';
    };

    modes.forEach((button) => button.addEventListener('click', () => {
      setPressed(modes, button);
      render(button.dataset.normMode);
    }));
    render('layer');
  });

  document.querySelectorAll('[data-projection-viz]').forEach((root) => {
    const values = { Q: [1, 2, 0, -1], K: [1, 2, -1, 0], V: [2, 2, 0, 1] };
    const modeButtons = [...root.querySelectorAll('[data-projection-mode]')];
    const rows = root.querySelector('[data-projection-rows]');
    const readout = root.querySelector('[data-projection-readout]');

    const render = (split) => {
      rows.innerHTML = Object.entries(values).map(([name, row]) => {
        const content = split
          ? '<div class="feature-strip"><span class="head-group"><span class="head-tag">H1</span>' + cells(row.slice(0, 2)) + '</span><span class="head-group"><span class="head-tag">H2</span>' + cells(row.slice(2)) + '</span></div>'
          : '<div class="feature-strip">' + cells(row) + '</div>';
        return '<div class="projection-row"><span class="projection-label">' + name + ' = xW<sub>' + name + '</sub></span>' + content + '</div>';
      }).join('');
      readout.textContent = split
        ? 'Only the feature axis changed: 4 features became 2 heads × 2 features. The token axis is still present.'
        : 'Each learned matrix produces a different four-feature view of the same token.';
    };

    modeButtons.forEach((button) => button.addEventListener('click', () => {
      setPressed(modeButtons, button);
      render(button.dataset.projectionMode === 'split');
    }));
    render(false);
  });

  document.querySelectorAll('[data-head-viz]').forEach((root) => {
    const tokens = ['The', 'cat', 'sat', '.'];
    const matrices = [
      [[1, 0, 0, 0], [0.42, 0.58, 0, 0], [0.25, 0.45, 0.30, 0], [0.10, 0.20, 0.30, 0.40]],
      [[1, 0, 0, 0], [0.75, 0.25, 0, 0], [0.15, 0.20, 0.65, 0], [0.05, 0.10, 0.25, 0.60]]
    ];
    const container = root.querySelector('[data-head-visuals]');
    const readout = root.querySelector('[data-head-readout]');
    let query = 2;

    const render = () => {
      container.innerHTML = matrices.map((matrix, head) => {
        const tokenButtons = tokens.map((token, index) =>
          '<button type="button" data-head-token="' + index + '" aria-pressed="' + (query === index) + '">' + token + '</button>'
        ).join('');
        const heat = matrix.flatMap((row, rowIndex) => row.map((weight) =>
          '<span class="attention-mini-cell ' + (rowIndex === query ? '' : 'inactive') + '" style="--weight:' + weight + '">' +
          (rowIndex === query ? weight.toFixed(2) : '') + '</span>'
        )).join('');
        return '<div class="head-visual"><h4>Head ' + (head + 1) + ' · all four tokens</h4><div class="token-strip">' + tokenButtons +
          '</div><div class="attention-mini-grid">' + heat + '</div></div>';
      }).join('');
      readout.textContent = 'Both heads use query “' + tokens[query] + '” over the same four keys; their learned weight patterns differ.';
      container.querySelectorAll('[data-head-token]').forEach((button) => button.addEventListener('click', () => {
        query = Number(button.dataset.headToken);
        render();
      }));
    };
    render();
  });

  document.querySelectorAll('[data-residual-viz]').forEach((root) => {
    const x = [1, -2, 0.5];
    const update = [0.2, 0.3, -0.1];
    const slider = root.querySelector('[data-update-scale]');
    const scaleReadout = root.querySelector('[data-update-scale-readout]');
    const xNode = root.querySelector('[data-residual-x]');
    const updateNode = root.querySelector('[data-residual-update]');
    const outputNode = root.querySelector('[data-residual-output]');
    const caption = root.querySelector('[data-residual-readout]');

    const render = () => {
      const scale = Number(slider.value);
      const scaled = update.map((value) => value * scale);
      const output = x.map((value, index) => value + scaled[index]);
      xNode.innerHTML = cells(x);
      updateNode.innerHTML = cells(scaled);
      outputNode.innerHTML = cells(output);
      scaleReadout.textContent = scale.toFixed(1);
      caption.textContent = 'The identity path stays fixed while the learned correction is scaled: Y = (' + output.map(format).join(', ') + ').';
    };
    slider.addEventListener('input', render);
    render();
  });

  document.querySelectorAll('[data-swiglu-viz]').forEach((root) => {
    const x1 = root.querySelector('[data-swiglu-x1]');
    const x2 = root.querySelector('[data-swiglu-x2]');
    const inputReadout = root.querySelector('[data-swiglu-input]');
    const gateNode = root.querySelector('[data-swiglu-gate]');
    const siluNode = root.querySelector('[data-swiglu-silu]');
    const upNode = root.querySelector('[data-swiglu-up]');
    const productNode = root.querySelector('[data-swiglu-product]');
    const outputNode = root.querySelector('[data-swiglu-output]');

    const sigmoid = (z) => 1 / (1 + Math.exp(-z));
    const renderPair = (node, values) => { node.innerHTML = values.map((x) => '<span>' + format(x) + '</span>').join(''); };
    const render = () => {
      const x = [Number(x1.value), Number(x2.value)];
      const gate = x;
      const silu = gate.map((z) => z * sigmoid(z));
      const up = [x[0] + x[1], x[0] - x[1]];
      const product = silu.map((value, index) => value * up[index]);
      const output = [product[0], -2 * product[1]];
      inputReadout.textContent = '(' + x.map(format).join(', ') + ')';
      renderPair(gateNode, gate);
      renderPair(siluNode, silu);
      renderPair(upNode, up);
      renderPair(productNode, product);
      renderPair(outputNode, output);
    };
    [x1, x2].forEach((input) => input.addEventListener('input', render));
    render();
  });

  document.querySelectorAll('[data-block-stepper]').forEach((root) => {
    const stages = [
      { name: 'Norm', input: 'X⁽ˡ⁾', operation: 'Normalize each token', output: 'X̄', shape: 'B × T × d_model → same shape' },
      { name: 'QKV', input: 'X̄', operation: 'Multiply by WQ, WK, WV', output: 'Q · K · V', shape: 'Three learned views; then split features into heads' },
      { name: 'RoPE', input: 'Q · K', operation: 'Rotate feature pairs', output: 'Q′ · K′', shape: 'Position changes phase; V is unchanged' },
      { name: 'Score', input: 'Q′ · K′', operation: 'Q′K′ᵀ / √dₕ + mask', output: 'A', shape: 'B × h × T × T query-key scores' },
      { name: 'Blend', input: 'softmax(A) · V', operation: 'Weighted value sum', output: 'O₁ … Oₕ', shape: 'Each head returns T × dₕ' },
      { name: 'Join', input: 'O₁ … Oₕ', operation: 'Concat, then Wₒ', output: 'O_attn', shape: 'Back to B × T × d_model' },
      { name: 'Add', input: 'X⁽ˡ⁾ + O_attn', operation: 'First residual', output: 'Y', shape: 'Identity state plus attention update' },
      { name: 'FFN', input: 'Y', operation: 'Norm → SwiGLU → add', output: 'X⁽ˡ⁺¹⁾', shape: 'Feature update; outer tensor shape preserved' }
    ];
    const strip = root.querySelector('[data-block-steps]');
    const input = root.querySelector('[data-stage-input]');
    const operation = root.querySelector('[data-stage-operation]');
    const output = root.querySelector('[data-stage-output]');
    const status = root.querySelector('[data-stage-status]');
    const previous = root.querySelector('[data-stage-previous]');
    const next = root.querySelector('[data-stage-next]');
    let stage = 0;

    strip.innerHTML = stages.map((item, index) =>
      '<button type="button" data-stage="' + index + '" aria-label="Stage ' + (index + 1) + ': ' + item.name + '">' + (index + 1) + '</button>'
    ).join('');

    const render = () => {
      const item = stages[stage];
      strip.querySelectorAll('[data-stage]').forEach((button, index) => {
        if (index === stage) button.setAttribute('aria-current', 'step');
        else button.removeAttribute('aria-current');
      });
      input.innerHTML = '<div><strong>' + item.input + '</strong><span>stage input</span></div>';
      operation.textContent = item.operation;
      output.innerHTML = '<div><strong>' + item.output + '</strong><span>stage output</span></div>';
      status.textContent = 'Stage ' + (stage + 1) + ' of ' + stages.length + ' · ' + item.shape;
      previous.disabled = stage === 0;
      next.disabled = stage === stages.length - 1;
    };
    strip.querySelectorAll('[data-stage]').forEach((button) => button.addEventListener('click', () => {
      stage = Number(button.dataset.stage);
      render();
    }));
    previous.addEventListener('click', () => { if (stage > 0) stage -= 1; render(); });
    next.addEventListener('click', () => { if (stage < stages.length - 1) stage += 1; render(); });
    render();
  });
})();
