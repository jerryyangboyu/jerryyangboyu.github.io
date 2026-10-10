(() => {
  const lessonId = document.body?.dataset.lesson;
  const lessons = {
    '0015': {
      title: 'Read a hidden state as a table of token vectors',
      retrieval: 'Without looking above, explain what X[b, t, :] selects and why the feature axis is the one that remains.',
      answer: 'Fixing a batch index b and token index t leaves every feature, so X[b,t,:] has shape [D]. The batch and sequence axes identify a location; D is the representation stored at that location.',
      bridge: 'The embedding lookup introduces D while preserving B and T. The rest of the block is shape-safe only if each sublayer eventually returns to [B,T,D].',
      code: `token_ids = torch.tensor([[4, 7, 2], [9, 1, 5]])  # [B=2, T=3]\nx = nn.Embedding(32, 8)(token_ids)                  # [B, T, D]\nassert x.shape == (2, 3, 8)`,
      task: 'Write select_token(x, batch, token) for x shaped [B,T,D]. Return one [D] vector and raise IndexError when either index is out of range.',
      solution: `def select_token(x, batch, token):\n    B, T, D = x.shape\n    if not (0 <= batch < B and 0 <= token < T):\n        raise IndexError("batch/token outside the tensor")\n    return x[batch, token, :]`,
      practice: ['For B=3,T=5,D=8, how many scalars are in X?', 'What is the shape of X[:,3,:]?', 'Why must a residual block return to D after an internal expansion?'],
      solutions: ['120', '[B,D] = [3,8]', 'Elementwise addition requires the update and state to have the same shape.']
    },
    '0016': {
      title: 'Normalize features independently for each token',
      retrieval: 'Which axis does RMSNorm reduce, and what does it leave unchanged?',
      answer: 'It reduces the last axis, D, independently for every [batch, token] pair. It rescales feature values but keeps the outer shape [B,T,D].',
      bridge: 'MiniMind uses RMSNorm before attention and before its MLP. The choice of reduction axis is a runtime contract, not a cosmetic detail: reducing over T would mix tokens.',
      code: `class RMSNorm(nn.Module):\n    def forward(self, x):\n        rms = torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True) + eps)\n        return self.weight * x * rms       # [B, T, D]`,
      task: 'Write a complete rms_norm(x, weight, eps) function. Reduce the feature axis and assert that a [2,3,8] input returns [2,3,8].',
      solution: `def rms_norm(x, weight, eps):\n    variance = x.float().pow(2).mean(dim=-1, keepdim=True)\n    scaled = x * torch.rsqrt(variance + eps).to(x.dtype)\n    output = weight * scaled\n    assert output.shape == x.shape\n    return output`,
      practice: ['Compute RMS(1,2,3).', 'State one difference between LayerNorm and RMSNorm.', 'Why is keepdim=True useful here?'],
      solutions: ['sqrt(14/3) ≈ 2.160', 'LayerNorm centers by subtracting a mean; RMSNorm only rescales.', 'It keeps a singleton feature axis so broadcasting is explicit and shape-safe.']
    },
    '0017': {
      title: 'Split feature width into heads without splitting the sentence',
      retrieval: 'With T=5, D=12, and H=3, what does one attention head receive?',
      answer: 'Every head sees all five token positions, but only Dh=D/H=4 features per position. Splitting is along feature width, not along the sequence axis.',
      bridge: 'Q and K receive RoPE after projection; V does not. A useful ledger is [B,T,D] → [B,H,T,Dh] before scores, then back to [B,T,D] after concatenation.',
      code: `q = self.q_proj(x).view(B, T, H, Dh).transpose(1, 2)\nk = self.k_proj(x).view(B, T, H, Dh).transpose(1, 2)\nv = self.v_proj(x).view(B, T, H, Dh).transpose(1, 2)\nq, k = apply_rotary_pos_emb(q, k)`,
      task: 'Write a complete split_heads(projected, n_heads) function. Turn [B,T,H*Dh] into [B,H,T,Dh] and assert the output shape.',
      solution: `def split_heads(projected, n_heads):\n    B, T, width = projected.shape\n    assert width % n_heads == 0\n    Dh = width // n_heads\n    heads = projected.view(B, T, n_heads, Dh).transpose(1, 2)\n    assert heads.shape == (B, n_heads, T, Dh)\n    return heads`,
      practice: ['Why does V stay unrotated?', 'What shape does one head’s score matrix have?', 'What must be true of D, H, and Dh?'],
      solutions: ['RoPE encodes relative position in query-key comparisons; V carries content.', '[T,T] per head and batch item.', 'D = H·Dh.']
    },
    '0018': {
      title: 'Mask future positions before probability is formed',
      retrieval: 'For a query at position i, which keys j are legal in a causal decoder?',
      answer: 'Only j ≤ i. The current token may attend to itself and earlier tokens, never to a later token.',
      bridge: 'The mask is added to scores before softmax. A forbidden score becomes −∞, its exponential becomes zero, and the row is normalized only over legal positions.',
      code: `scores = (q @ k.transpose(-2, -1)) / math.sqrt(Dh)\nscores = scores.masked_fill(~causal_mask, torch.finfo(scores.dtype).min)\nweights = torch.softmax(scores.float(), dim=-1)`,
      task: 'Write a complete causal_softmax(scores) function. Mask future positions before softmax and test that the upper triangle is zero.',
      solution: `def causal_softmax(scores):\n    T = scores.shape[-1]\n    mask = torch.tril(torch.ones(T, T, dtype=torch.bool, device=scores.device))\n    masked = scores.masked_fill(~mask, torch.finfo(scores.dtype).min)\n    probs = torch.softmax(masked.float(), dim=-1).to(scores.dtype)\n    assert torch.allclose(probs.triu(1), torch.zeros_like(probs.triu(1)))\n    return probs`,
      practice: ['For i=2, list the legal keys.', 'Why would masking after softmax be wrong?', 'What does the diagonal mean?'],
      solutions: ['j=0,1,2 (or 1,2,3 with one-based notation).', 'Future positions would receive positive probability before they were removed.', 'A token may read its own content.']
    },
    '0019': {
      title: 'Turn comparisons into a weighted content read',
      retrieval: 'What invariant must every attention probability row satisfy?',
      answer: 'Each row sums to one. The weights are a distribution over allowed key/value positions, so the output is a weighted average of value vectors.',
      bridge: 'The score-key axis disappears in weights @ V. For weights [B,H,Tq,Tk] and values [B,H,Tk,Dh], context is [B,H,Tq,Dh].',
      code: `weights = F.softmax(scores.float(), dim=-1).to(q.dtype)\nassert torch.allclose(weights.float().sum(dim=-1), torch.ones_like(weights[..., 0].float()))\ncontext = weights @ v                       # [B,H,Tq,Dh]`,
      task: 'Write a complete attention_values(scores, values) function. Compute stable row-wise probabilities, check their sums, and return the weighted values.',
      solution: `def attention_values(scores, values):\n    weights = torch.softmax(scores.float(), dim=-1).to(values.dtype)\n    row_sums = weights.float().sum(dim=-1)\n    assert torch.allclose(row_sums, torch.ones_like(row_sums), atol=1e-5)\n    return weights @ values`,
      practice: ['Compute softmax(0,0).', 'With weights (0.25,0.75) and values (2,0),(0,4), compute the output.', 'Why calculate softmax in float32?'],
      solutions: ['(0.5,0.5)', '(0.5,3)', 'To reduce overflow, underflow, and precision loss in exponentials.']
    },
    '0020': {
      title: 'Compress K/V heads while preserving Q’s comparisons',
      retrieval: 'What changes in grouped-query attention when H=8 and Hkv=2?',
      answer: 'Queries keep eight heads. Keys and values have two heads, each repeated H/Hkv=4 times so every query head has a matching K/V head.',
      bridge: 'GQA saves the cache memory used by K/V while retaining many query subspaces. The divisibility rule is a contract: H % Hkv must equal zero.',
      code: `def repeat_kv(x, n_rep):\n    # x: [B, Hkv, T, Dh]\n    return x[:, :, None, :, :].expand(B, Hkv, n_rep, T, Dh)\n        .reshape(B, Hkv * n_rep, T, Dh)`,
      task: 'Write a complete repeat_kv(kv, n_query_heads) function. Expand the K/V heads while keeping T and Dh unchanged.',
      solution: `def repeat_kv(kv, n_query_heads):\n    B, Hkv, T, Dh = kv.shape\n    assert n_query_heads % Hkv == 0\n    repeats = n_query_heads // Hkv\n    expanded = kv[:, :, None].expand(B, Hkv, repeats, T, Dh)\n    return expanded.reshape(B, n_query_heads, T, Dh)`,
      practice: ['Why does GQA not change T?', 'What is cached during decoding?', 'What failure should configuration validation catch?'],
      solutions: ['It groups feature heads, not token positions.', 'Past K and V states, indexed by token position.', 'Reject H values not divisible by Hkv.']
    },
    '0021': {
      title: 'Treat the residual stream as the state contract',
      retrieval: 'What must be true before an update is added to the residual stream?',
      answer: 'The update must have the same [B,T,D] shape as the state. Attention and the MLP write corrections; they do not replace the state.',
      bridge: 'The identity path carries the current representation directly through each sublayer. This gives the model a stable state and gives gradients a short route.',
      code: `residual = hidden_states\nhidden_states = self.self_attn(self.input_layernorm(hidden_states), ...)\nhidden_states = hidden_states + residual\nhidden_states = hidden_states + self.mlp(self.post_attention_layernorm(hidden_states))`,
      task: 'Write a complete residual_block(x, attention_update, mlp_update) function. Check both updates before adding them to the state.',
      solution: `def residual_block(x, attention_update, mlp_update):\n    assert attention_update.shape == x.shape\n    state = x + attention_update\n    assert mlp_update.shape == state.shape\n    return state + mlp_update`,
      practice: ['Write the first residual equation.', 'Where does attention mix tokens?', 'Where does the MLP mix features?'],
      solutions: ['Y = X + O_attn', 'In the weighted value read across legal positions.', 'Within each token row, independently across D.']
    },
    '0022': {
      title: 'Make the SwiGLU width changes explicit',
      retrieval: 'What must match before the Hadamard product SiLU(g) ⊙ u?',
      answer: 'The gate and up branches must have the same [B,T,Dff] shape. The down projection then returns the update to [B,T,D].',
      bridge: 'SwiGLU transforms features independently for each token. It is the second learned write in the pre-norm block; it does not mix information across token positions.',
      code: `gate = F.silu(self.gate_proj(x))  # [B,T,Dff]\nup = self.up_proj(x)              # [B,T,Dff]\ngated = gate * up\nreturn self.down_proj(gated)      # [B,T,D]`,
      task: 'Write a complete swiglu(x, gate_proj, up_proj, down_proj) function. Assert the two expanded branches agree before multiplying.',
      solution: `def swiglu(x, gate_proj, up_proj, down_proj):\n    gate = F.silu(gate_proj(x))\n    up = up_proj(x)\n    assert gate.shape == up.shape\n    return down_proj(gate * up)`,
      practice: ['Compute SiLU(0).', 'For gate (0.5,-1) and up (4,3), compute the product.', 'Which stage restores model width?'],
      solutions: ['0', '(2,-3)', 'The down projection.']
    },
    '0023': {
      title: 'Teach one complete MiniMindBlock without skipping a contract',
      retrieval: 'Name the two learned writes and the state that carries between them.',
      answer: 'Attention writes a token-mixing update \\(O_{\\mathrm{attn}}\\); the MLP writes a per-token feature update \\(F\\). The residual stream carries \\(X\\to Y\\to X^{(\\ell+1)}\\), while the attention cache carries past K/V for decoding.',
      bridge: 'Read the complete MiniMindBlock.forward function as one state transition: normalize before attention, add the attention update, normalize before the MLP, add the feature update, then return the state and cache. Every state remains \\(B\\times T\\times D\\).',
      code: `def forward(\n    self,\n    hidden_states,\n    position_embeddings=None,\n    past_key_value=None,\n    use_cache=False,\n    attention_mask=None,\n):\n    residual = hidden_states\n    normalized = self.input_layernorm(hidden_states)\n    hidden_states, present_key_value = self.self_attn(\n        normalized,\n        position_embeddings,\n        past_key_value,\n        use_cache,\n        attention_mask,\n    )\n    assert hidden_states.shape == residual.shape\n    hidden_states = hidden_states + residual\n    normalized = self.post_attention_layernorm(hidden_states)\n    mlp_update = self.mlp(normalized)\n    assert mlp_update.shape == residual.shape\n    hidden_states = hidden_states + mlp_update\n    return hidden_states, present_key_value`,
      task: 'Implement a complete MiniMindBlock module. Write __init__ with its two RMSNorm layers, attention, and MLP, then write forward with shape assertions and a cached decode case where \\(T_q=1\\).',
      solution: `class MiniMindBlock(nn.Module):\n    def __init__(self, config):\n        super().__init__()\n        self.input_layernorm = RMSNorm(config.hidden_size, config.rms_norm_eps)\n        self.self_attn = MiniMindAttention(config)\n        self.post_attention_layernorm = RMSNorm(config.hidden_size, config.rms_norm_eps)\n        self.mlp = FeedForward(config)\n\n    def forward(\n        self, hidden_states, position_embeddings=None,\n        past_key_value=None, use_cache=False, attention_mask=None\n    ):\n        assert hidden_states.ndim == 3\n        residual = hidden_states\n        hidden_states, present = self.self_attn(\n            self.input_layernorm(hidden_states),\n            position_embeddings, past_key_value,\n            use_cache, attention_mask\n        )\n        assert hidden_states.shape == residual.shape\n        hidden_states = hidden_states + residual\n        mlp_out = self.mlp(self.post_attention_layernorm(hidden_states))\n        assert mlp_out.shape == residual.shape\n        hidden_states = hidden_states + mlp_out\n        return hidden_states, present`,
      practice: ['Where does token information mix?', 'Where does feature information mix?', 'What is allowed to grow during cached decoding?'],
      solutions: ['In masked attention’s weighted sum over earlier values.', 'In Q/K/V and output projections, and in the MLP, one token at a time.', 'The cache’s key/value token axis; the returned hidden state keeps the requested query length.']
    }
  };
  const walkthrough = {
    '0015': ['Create a small batch of integer token IDs.', 'Look up one D-dimensional vector for each token.', 'Check the complete [B,T,D] contract.'],
    '0016': ['Define the normalization module and its forward method.', 'Reduce only across features, keeping the axis for broadcasting.', 'Scale the original features; the outer shape is unchanged.'],
    '0017': ['Project the model state and expose H heads with Dh features.', 'Apply the same ledger to keys.', 'Apply it to values; these carry content rather than position.', 'Rotate Q and K before forming their dot products.'],
    '0018': ['Form scaled query-key comparisons.', 'Replace future scores before probabilities are computed.', 'Softmax each legal score row in a stable dtype.'],
    '0019': ['Compute probabilities in float32, then return to the model dtype.', 'Check the probability invariant explicitly.', 'Use the probabilities to average value vectors.'],
    '0020': ['Define a helper for expanding grouped K/V heads.', 'Document the compact K/V shape.', 'Insert a repeat axis without copying values.', 'Flatten the repeat axis into the full query-head axis.'],
    '0021': ['Keep a copy of the state that will receive the attention update.', 'Normalize before attention; the sublayer returns an update.', 'Write the update into the state without changing its shape.', 'Normalize the new state, then write the MLP update.'],
    '0022': ['The gate branch learns a feature-wise control signal.', 'The up branch produces a matching expanded representation.', 'Combine the two branches coordinate by coordinate.', 'Project the expanded result back to model width.'],
    '0023': ['Complete state transition.', '', 'Input state [B,T,D].', '', 'Reuse past K/V during decoding.', '', '', '', 'Save the identity path.', 'Normalize before attention.', 'Attention returns a token-mixing update and cache.', '', '', '', '', '', '', 'Shape check: update stays [B,T,D].', 'First residual write.', 'Normalize before the MLP.', 'MLP writes the feature update.', 'Shape check: update stays [B,T,D].', 'Second residual write.', 'Return the state and cache.']
  };
  const task = lessons[lessonId];
  if (!task) return;
  const esc = (value) => String(value).replace(/[&<>"']/g, (char) => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' }[char]));
  const lines = task.code.split('\n');
  const notes = walkthrough[lessonId] || [];
  const questions = [task.retrieval, ...task.practice];
  const answers = [task.answer, ...task.solutions];
  const walkthroughHtml = lines.map((line, index) => `<div class="code-line"><code class="language-python">${esc(line || ' ')}</code><span>${esc(notes[index] || '')}</span></div>`).join('');
  document.querySelectorAll('.practice-studio').forEach((host) => {
    const sequence = document.createElement('section');
    sequence.className = 'learning-sequence';
    sequence.id = 'practice-before-moving-on';
    sequence.innerHTML = `<h2>Practice before moving on</h2><section class="learning-part"><h3>Code walkthrough</h3><p>${esc(task.bridge)}</p><div class="code-walkthrough">${walkthroughHtml}</div></section><section id="code-checkpoint" class="learning-part"><h3>Coding challenge</h3><p>${esc(task.task)}</p><details><summary>Reference implementation</summary><pre><code class="language-python">${esc(task.solution)}</code></pre><p class="answer-note">Compare the assertions and shapes with your own implementation.</p></details></section><section class="learning-part"><h3>Exercises</h3><ol>${questions.map((item) => `<li>${esc(item)}</li>`).join('')}</ol><details><summary>Answers</summary><ol>${answers.map((item) => `<li>${esc(item)}</li>`).join('')}</ol></details></section>`;
    host.prepend(sequence);
    const highlight = () => {
      if (!window.hljs) return;
      sequence.querySelectorAll('pre code[class*="language-"], .code-line code[class*="language-"]').forEach((code) => window.hljs.highlightElement(code));
    };
    highlight();
    window.addEventListener('load', highlight, { once: true });
    const typeset = () => window.MathJax?.typesetPromise?.([sequence]);
    if (window.MathJax?.typesetPromise) typeset();
    else window.addEventListener('load', typeset, { once: true });
    const cards = [...host.children].filter((node) => node.classList.contains('studio-card'));
    cards.forEach((card) => card.remove());
  });
})();
