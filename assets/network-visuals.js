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
    root.classList.add('viz-tensor');
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
    root.classList.add('viz-normalize');
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
    root.classList.add('viz-projection');
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
    root.classList.add('viz-heads');
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
    root.classList.add('viz-residual');
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
    root.classList.add('viz-swiglu');
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

    // Connect the actual rendered nodes so branches remain attached when the
    // diagram reflows or MathJax/fonts change its dimensions.
    const graph = root.querySelector('.swiglu-graph');
    const wires = document.createElementNS('http://www.w3.org/2000/svg', 'svg');
    wires.classList.add('swiglu-wires');
    wires.setAttribute('aria-hidden', 'true');
    graph.prepend(wires);
    const drawWires = () => {
      const bounds = graph.getBoundingClientRect();
      const compact = matchMedia('(max-width: 900px)').matches;
      const anchor = (element, side) => {
        const r = element.getBoundingClientRect();
        return { x: (side === 'left' ? r.left : side === 'right' ? r.right : r.left + r.width / 2) - bounds.left,
          y: (side === 'top' ? r.top : side === 'bottom' ? r.bottom : r.top + r.height / 2) - bounds.top };
      };
      const input = root.querySelector('.swiglu-input-node');
      const gate = gateNode.closest('.swiglu-node');
      const silu = siluNode.closest('.swiglu-node');
      const up = upNode.closest('.swiglu-node');
      const merge = root.querySelector('.swiglu-merge span');
      const product = productNode.closest('.swiglu-node');
      const output = outputNode.closest('.swiglu-node');
      const paths = [];
      const connect = (a, b, route = 'horizontal') => {
        const middle = (a.x + b.x) / 2;
        const y = (a.y + b.y) / 2;
        paths.push(route === 'vertical' ? `M${a.x},${a.y} V${y} H${b.x} V${b.y}`
          : route === 'outside' ? `M${a.x},${a.y} H${a.x - 12} V${b.y} H${b.x}`
          : `M${a.x},${a.y} H${middle} V${b.y} H${b.x}`);
      };
      connect(anchor(input, 'right'), anchor(gate, 'left'));
      connect(anchor(input, 'right'), anchor(up, 'left'));
      connect(anchor(gate, 'bottom'), anchor(silu, 'top'), 'vertical');
      if (compact) {
        connect(anchor(silu, 'left'), anchor(merge, 'left'), 'outside');
        connect(anchor(up, 'bottom'), anchor(merge, 'top'), 'vertical');
        // Leave the product label beneath the circle unobstructed.
        connect(anchor(root.querySelector('.swiglu-merge'), 'bottom'), anchor(product, 'top'), 'vertical');
        connect(anchor(product, 'bottom'), anchor(output, 'top'), 'vertical');
      } else {
        connect(anchor(silu, 'right'), anchor(merge, 'left'));
        connect(anchor(up, 'right'), anchor(merge, 'left'));
        connect(anchor(merge, 'right'), anchor(product, 'left'));
        connect(anchor(product, 'right'), anchor(output, 'left'));
      }
      wires.setAttribute('viewBox', `0 0 ${bounds.width} ${bounds.height}`);
      wires.innerHTML = '<defs><marker id="swiglu-direction" markerWidth="5" markerHeight="5" refX="5" refY="2.5" orient="auto"><polygon points="0,0 5,2.5 0,5" fill="var(--muted)"/></marker></defs>' +
        paths.map(d => `<path d="${d}" marker-end="url(#swiglu-direction)"/>`).join('');
    };
    const wireObserver = new ResizeObserver(drawWires);
    wireObserver.observe(graph);
    graph.querySelectorAll('.swiglu-node, .swiglu-input-node, .swiglu-merge').forEach(node => wireObserver.observe(node));
    document.fonts.ready.then(drawWires);
  });

  document.querySelectorAll('[data-block-explorer]').forEach((figure) => {
    const explorer = figure.querySelector('[data-stage-detail]');
    if (!explorer) return;

    const stages = {
      whole: {
        title: 'MiniMindBlock.forward', index: 'model/model_minimind.py · block',
        summary: 'The exact MiniMind execution order: save residual, call self_attn on input_layernorm(hidden_states), add it back, call self.mlp on post_attention_layernorm, add the second update, and return the state with present_key_value.',
        equation: '\\(Y=X^{(\\ell)}+O_{\\mathrm{attn}},\\qquad X^{(\\ell+1)}=Y+\\operatorname{SwiGLU}(\\operatorname{Norm}(Y))\\)',
        source: 'MiniMind · model/model_minimind.py · MiniMindBlock.forward', href: '../reference/transformer-model-code.html',
        code: `residual = hidden_states
hidden_states, present_key_value = self.self_attn(
    self.input_layernorm(hidden_states), position_embeddings,
    past_key_value, use_cache, attention_mask
)
hidden_states += residual
hidden_states = hidden_states + self.mlp(
    self.post_attention_layernorm(hidden_states)
)`
      },
      input: {
        title: 'Input state', index: 'step 0 · residual stream',
        summary: 'The block receives one hidden vector per token. Every later operation must preserve the batch, token, and model axes.',
        equation: '\\(X^{(\\ell)}\\in\\mathbb{R}^{B\\times T\\times D}\\)',
        source: 'MiniMind · model/model_minimind.py · MiniMindModel', href: '../reference/transformer-model-code.html',
        code: `hidden_states = self.embed_tokens(input_ids)
# shape: [batch, tokens, hidden_size]`
      },
      attention: {
        title: 'Attention write', index: 'step 1 · mix tokens',
        summary: 'Attention is the cross-token branch: normalize each row, compare queries with keys, blend values, then project the joined heads back to D features.',
        equation: '\\(O_{\\mathrm{attn}}=\\operatorname{Concat}(O_1,\\ldots,O_H)W_O\\)',
        source: 'MiniMind · model/model_minimind.py · Attention.forward', href: '../reference/transformer-model-code.html#attention',
        code: `xq, xk, xv = self.q_proj(x), self.k_proj(x), self.v_proj(x)
output = self.o_proj(attn_output)
# returns [batch, tokens, hidden_size]`
      },
      rope: {
        title: 'RoPE on Q and K', index: 'step 2 · encode position',
        summary: 'Rotary position embeddings rotate query and key pairs by token position. Values keep their content coordinates; only comparisons acquire relative position information.',
        equation: '\\(Q\\prime=\\operatorname{RoPE}(Q),\\qquad K\\prime=\\operatorname{RoPE}(K)\\)',
        source: 'MiniMind · model/model_minimind.py · rotary embedding', href: '../reference/transformer-model-code.html#rope',
        code: `xq, xk = apply_rotary_pos_emb(
    xq, xk, cos, sin
)
# xv is unchanged by RoPE`
      },
      mask: {
        title: 'Causal mask', index: 'step 3 · compare and mask',
        summary: 'Scaled query-key scores become probabilities only after the future-token entries are set to −∞. The triangular pattern is the causal contract.',
        equation: '\\(P=\\operatorname{softmax}\\!\\left(\\frac{Q\\prime K\\prime^{\\mathsf T}}{\\sqrt{d_h}}+M_{\\mathrm{causal}}\\right)\\)',
        source: 'MiniMind · model/model_minimind.py · score path', href: '../reference/transformer-model-code.html#attention',
        code: `scores = torch.matmul(xq, xk.transpose(-2, -1)) * scale
scores = scores + causal_mask
probs = F.softmax(scores, dim=-1, dtype=torch.float32).type_as(xq)`
      },
      residual1: {
        title: 'First residual write', index: 'step 4 · preserve the stream',
        summary: 'The attention result is an update, not a replacement. Adding it to the original stream keeps an identity route for optimization and information flow.',
        equation: '\\(Y=X^{(\\ell)}+O_{\\mathrm{attn}}\\)',
        source: 'MiniMind · model/model_minimind.py · residual path', href: '../reference/transformer-model-code.html#block',
        code: `residual = hidden_states
hidden_states = self.self_attn(normed_states, ...)[0]
hidden_states += residual
# shape remains [B, T, D]`
      },
      ffn: {
        title: 'SwiGLU write', index: 'step 5 · mix features per token',
        summary: 'The feed-forward branch works independently on each token row. Its gated expansion mixes features, contracts back to D, and writes a second correction.',
        equation: '\\(F=W_{\\mathrm{down}}\\!\\left(\\operatorname{SiLU}(W_{\\mathrm{gate}}Y)\\odot W_{\\mathrm{up}}Y\\right)\\)',
        source: 'MiniMind · model/model_minimind.py · MLP path', href: '../reference/transformer-model-code.html#mlp',
        code: `hidden_states = hidden_states + self.mlp(
    self.post_attention_layernorm(hidden_states)
)
# dense SwiGLU or optional MoEFeedForward`
      },
      residual2: {
        title: 'Second residual write', index: 'step 6 · block output',
        summary: 'The second update completes the block. The output keeps the same [B, T, D] shape and becomes the input to the next decoder block.',
        equation: '\\(X^{(\\ell+1)}=Y+F\\)',
        source: 'MiniMind · model/model_minimind.py · MiniMindBlock.forward', href: '../reference/transformer-model-code.html#block',
        code: `hidden_states = hidden_states + self.mlp(
    self.post_attention_layernorm(hidden_states)
)
return hidden_states, present_key_value`
      }
    };

    const nodes = [...figure.querySelectorAll('[data-stage-node]')];
    const reset = explorer.querySelector('[data-stage-reset]');
    const title = explorer.querySelector('[data-stage-detail-title]');
    const index = explorer.querySelector('[data-stage-detail-index]');
    const summary = explorer.querySelector('[data-stage-detail-summary]');
    const equation = explorer.querySelector('[data-stage-detail-equation]');
    const source = explorer.querySelector('[data-stage-detail-source]');
    const sourceLink = explorer.querySelector('[data-stage-detail-link]');
    const code = explorer.querySelector('[data-stage-detail-code]');
    const codePanel = explorer.querySelector('[data-stage-code-panel]');
    let selectedNode = null;
    explorer.id = 'block-code-trace';

    const renderStage = (stageId) => {
      const stage = stages[stageId] || stages.whole;
      nodes.forEach((node) => {
        node.classList.toggle('is-active', stageId !== 'whole' && node.dataset.stageNode === stageId);
        node.setAttribute('aria-expanded', String(stageId !== 'whole' && node.dataset.stageNode === stageId));
        node.setAttribute('aria-controls', explorer.id);
      });
      figure.classList.toggle('has-stage-selection', stageId !== 'whole');
      if (reset) reset.hidden = stageId === 'whole';
      if (codePanel) codePanel.hidden = stageId === 'whole';
      title.textContent = stage.title;
      index.textContent = stage.index;
      summary.textContent = stage.summary;
      if (window.MathJax?.typesetClear) window.MathJax.typesetClear([equation]);
      equation.innerHTML = stage.equation;
      source.textContent = stage.source;
      sourceLink.href = stage.href;
      code.textContent = stage.code;
      code.removeAttribute('data-highlighted');
      if (window.hljs) window.hljs.highlightElement(code);
      if (window.MathJax && window.MathJax.typesetPromise) window.MathJax.typesetPromise([equation]).catch(() => {});
    };

    const closeTrace = () => {
      renderStage('whole');
      selectedNode?.focus({ preventScroll: true });
    };
    if (reset) reset.addEventListener('click', closeTrace);
    document.addEventListener('keydown', (event) => {
      if (event.key === 'Escape' && figure.classList.contains('has-stage-selection')) {
        event.preventDefault();
        closeTrace();
      }
    });
    nodes.forEach((node) => {
      const openTrace = () => {
        selectedNode = node;
        renderStage(node.dataset.stageNode);
        reset?.focus({ preventScroll: true });
        if (window.matchMedia('(max-width: 720px)').matches) explorer.scrollIntoView({ block: 'nearest' });
      };
      node.addEventListener('click', openTrace);
      node.addEventListener('keydown', (event) => {
        if (event.key === 'Enter' || event.key === ' ') {
          event.preventDefault();
          openTrace();
        }
      });
    });
    renderStage('whole');
  });

})();
