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
      task: 'Implement RMSNorm with a reduction over the feature axis. Add a test that a [2,3,8] input returns [2,3,8].',
      solution: `def forward(self, x):\n    variance = x.float().pow(2).mean(dim=-1, keepdim=True)\n    x = x * torch.rsqrt(variance + self.eps).to(x.dtype)\n    return self.weight * x\n\nassert norm(torch.zeros(2, 3, 8)).shape == (2, 3, 8)`,
      practice: ['Compute RMS(1,2,3).', 'State one difference between LayerNorm and RMSNorm.', 'Why is keepdim=True useful here?'],
      solutions: ['sqrt(14/3) ≈ 2.160', 'LayerNorm centers by subtracting a mean; RMSNorm only rescales.', 'It keeps a singleton feature axis so broadcasting is explicit and shape-safe.']
    },
    '0017': {
      title: 'Split feature width into heads without splitting the sentence',
      retrieval: 'With T=5, D=12, and H=3, what does one attention head receive?',
      answer: 'Every head sees all five token positions, but only Dh=D/H=4 features per position. Splitting is along feature width, not along the sequence axis.',
      bridge: 'Q and K receive RoPE after projection; V does not. A useful ledger is [B,T,D] → [B,H,T,Dh] before scores, then back to [B,T,D] after concatenation.',
      code: `q = self.q_proj(x).view(B, T, H, Dh).transpose(1, 2)\nk = self.k_proj(x).view(B, T, H, Dh).transpose(1, 2)\nv = self.v_proj(x).view(B, T, H, Dh).transpose(1, 2)\nq, k = apply_rotary_pos_emb(q, k)`,
      task: 'Write the reshape and transpose that turns a [B,T,H*Dh] projection into [B,H,T,Dh]. Assert the output shape.',
      solution: `projected = projected.view(B, T, H, Dh).transpose(1, 2)\nassert projected.shape == (B, H, T, Dh)`,
      practice: ['Why does V stay unrotated?', 'What shape does one head’s score matrix have?', 'What must be true of D, H, and Dh?'],
      solutions: ['RoPE encodes relative position in query-key comparisons; V carries content.', '[T,T] per head and batch item.', 'D = H·Dh.']
    },
    '0018': {
      title: 'Mask future positions before probability is formed',
      retrieval: 'For a query at position i, which keys j are legal in a causal decoder?',
      answer: 'Only j ≤ i. The current token may attend to itself and earlier tokens, never to a later token.',
      bridge: 'The mask is added to scores before softmax. A forbidden score becomes −∞, its exponential becomes zero, and the row is normalized only over legal positions.',
      code: `scores = (q @ k.transpose(-2, -1)) / math.sqrt(Dh)\nscores = scores.masked_fill(~causal_mask, torch.finfo(scores.dtype).min)\nweights = torch.softmax(scores.float(), dim=-1)`,
      task: 'Implement a causal mask for T tokens and test that every entry above the diagonal receives zero probability after softmax.',
      solution: `mask = torch.tril(torch.ones(T, T, dtype=torch.bool, device=scores.device))\nscores = scores.masked_fill(~mask, torch.finfo(scores.dtype).min)\nprobs = torch.softmax(scores.float(), dim=-1)\nassert torch.allclose(probs.triu(1), torch.zeros_like(probs.triu(1)))`,
      practice: ['For i=2, list the legal keys.', 'Why would masking after softmax be wrong?', 'What does the diagonal mean?'],
      solutions: ['j=0,1,2 (or 1,2,3 with one-based notation).', 'Future positions would receive positive probability before they were removed.', 'A token may read its own content.']
    },
    '0019': {
      title: 'Turn comparisons into a weighted content read',
      retrieval: 'What invariant must every attention probability row satisfy?',
      answer: 'Each row sums to one. The weights are a distribution over allowed key/value positions, so the output is a weighted average of value vectors.',
      bridge: 'The score-key axis disappears in weights @ V. For weights [B,H,Tq,Tk] and values [B,H,Tk,Dh], context is [B,H,Tq,Dh].',
      code: `weights = F.softmax(scores.float(), dim=-1).to(q.dtype)\nassert torch.allclose(weights.float().sum(dim=-1), torch.ones_like(weights[..., 0].float()))\ncontext = weights @ v                       # [B,H,Tq,Dh]`,
      task: 'Write a numerically stable row-wise softmax and blend V. Check the row-sum invariant before casting back to the query dtype.',
      solution: `weights = torch.softmax(scores.float(), dim=-1).to(q.dtype)\nrow_sums = weights.float().sum(dim=-1)\nassert torch.allclose(row_sums, torch.ones_like(row_sums), atol=1e-5)\ncontext = weights @ value_states`,
      practice: ['Compute softmax(0,0).', 'With weights (0.25,0.75) and values (2,0),(0,4), compute the output.', 'Why calculate softmax in float32?'],
      solutions: ['(0.5,0.5)', '(0.5,3)', 'To reduce overflow, underflow, and precision loss in exponentials.']
    },
    '0020': {
      title: 'Compress K/V heads while preserving Q’s comparisons',
      retrieval: 'What changes in grouped-query attention when H=8 and Hkv=2?',
      answer: 'Queries keep eight heads. Keys and values have two heads, each repeated H/Hkv=4 times so every query head has a matching K/V head.',
      bridge: 'GQA saves the cache memory used by K/V while retaining many query subspaces. The divisibility rule is a contract: H % Hkv must equal zero.',
      code: `def repeat_kv(x, n_rep):\n    # x: [B, Hkv, T, Dh]\n    return x[:, :, None, :, :].expand(B, Hkv, n_rep, T, Dh)\n        .reshape(B, Hkv * n_rep, T, Dh)`,
      task: 'Implement repeat_kv and assert that the expanded K/V head count equals H while T and Dh stay unchanged.',
      solution: `n_rep = H // Hkv\nexpanded = kv[:, :, None].expand(B, Hkv, n_rep, T, Dh)\nexpanded = expanded.reshape(B, H, T, Dh)\nassert expanded.shape == (B, H, T, Dh)`,
      practice: ['Why does GQA not change T?', 'What is cached during decoding?', 'What failure should configuration validation catch?'],
      solutions: ['It groups feature heads, not token positions.', 'Past K and V states, indexed by token position.', 'Reject H values not divisible by Hkv.']
    },
    '0021': {
      title: 'Treat the residual stream as the state contract',
      retrieval: 'What must be true before an update is added to the residual stream?',
      answer: 'The update must have the same [B,T,D] shape as the state. Attention and the MLP write corrections; they do not replace the state.',
      bridge: 'The identity path carries the current representation directly through each sublayer. This gives the model a stable state and gives gradients a short route.',
      code: `residual = hidden_states\nhidden_states = self.self_attn(self.input_layernorm(hidden_states), ...)\nhidden_states = hidden_states + residual\nhidden_states = hidden_states + self.mlp(self.post_attention_layernorm(hidden_states))`,
      task: 'Add assertions around both residual writes in MiniMindBlock.forward. Make the test fail if a sublayer returns the wrong model width.',
      solution: `residual = hidden_states\nattn_out, present = self.self_attn(...)\nassert attn_out.shape == residual.shape\nhidden_states = residual + attn_out\nmlp_out = self.mlp(self.post_attention_layernorm(hidden_states))\nassert mlp_out.shape == hidden_states.shape\nhidden_states = hidden_states + mlp_out`,
      practice: ['Write the first residual equation.', 'Where does attention mix tokens?', 'Where does the MLP mix features?'],
      solutions: ['Y = X + O_attn', 'In the weighted value read across legal positions.', 'Within each token row, independently across D.']
    },
    '0022': {
      title: 'Make the SwiGLU width changes explicit',
      retrieval: 'What must match before the Hadamard product SiLU(g) ⊙ u?',
      answer: 'The gate and up branches must have the same [B,T,Dff] shape. The down projection then returns the update to [B,T,D].',
      bridge: 'SwiGLU transforms features independently for each token. It is the second learned write in the pre-norm block; it does not mix information across token positions.',
      code: `gate = F.silu(self.gate_proj(x))  # [B,T,Dff]\nup = self.up_proj(x)              # [B,T,Dff]\ngated = gate * up\nreturn self.down_proj(gated)      # [B,T,D]`,
      task: 'Implement dense SwiGLU with gate, up, elementwise product, and down stages. Assert that gate and up agree before multiplying.',
      solution: `gate = F.silu(self.gate_proj(x))\nup = self.up_proj(x)\nassert gate.shape == up.shape\nreturn self.down_proj(gate * up)`,
      practice: ['Compute SiLU(0).', 'For gate (0.5,-1) and up (4,3), compute the product.', 'Which stage restores model width?'],
      solutions: ['0', '(2,-3)', 'The down projection.']
    },
    '0023': {
      title: 'Teach one complete MiniMindBlock without skipping a contract',
      retrieval: 'Name the two learned writes and the state that carries between them.',
      answer: 'Attention writes a token-mixing update O_attn; the MLP writes a per-token feature update F. The residual stream carries X → Y → X(next), while the attention cache carries past K/V for decoding.',
      bridge: 'Read MiniMindBlock.forward as one sentence: normalize and attend, add the attention update, normalize and run the MLP, add the feature update, then return the state and cache. Every state remains [B,T,D].',
      code: `residual = hidden_states\nhidden_states, present_key_value = self.self_attn(\n    self.input_layernorm(hidden_states), position_embeddings,\n    past_key_value, use_cache, attention_mask\n)\nhidden_states = hidden_states + residual\nhidden_states = hidden_states + self.mlp(\n    self.post_attention_layernorm(hidden_states)\n)\nreturn hidden_states, present_key_value`,
      task: 'Instrument MiniMindBlock.forward with a shape ledger. Record [B,T,D] at entry, after attention, after both residuals, and at return. Add a cached decode case where Tq=1.',
      solution: `assert hidden_states.ndim == 3\nresidual = hidden_states\nhidden_states, present = self.self_attn(...)\nassert hidden_states.shape == residual.shape\nhidden_states = hidden_states + residual\nmlp_out = self.mlp(self.post_attention_layernorm(hidden_states))\nassert mlp_out.shape == residual.shape\nhidden_states = hidden_states + mlp_out\nreturn hidden_states, present`,
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
    '0023': ['Keep the incoming state for the first residual addition.', 'Call attention on the normalized state and receive its cache.', 'Pass positions and the decoding context into attention.', 'Supply the previous cache, cache flag, and mask.', 'Close the attention call.', 'Restore the residual state after the attention update.', 'Normalize that state before the MLP.', 'The MLP writes a second [B,T,D] update.', 'Close the MLP call.', 'Return the state and the new K/V cache.']
  };
  const task = lessons[lessonId];
  if (!task) return;
  const esc = (value) => String(value).replace(/[&<>"']/g, (char) => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' }[char]));
  const lines = task.code.split('\n');
  const notes = walkthrough[lessonId] || [];
  const questions = [task.retrieval, ...task.practice];
  const answers = [task.answer, ...task.solutions];
  const walkthroughHtml = lines.map((line, index) => `<div class="code-line"><code>${esc(line || ' ')}</code><span>${esc(notes[index] || '')}</span></div>`).join('');
  document.querySelectorAll('.practice-studio').forEach((host) => {
    const sequence = document.createElement('section');
    sequence.className = 'learning-sequence';
    sequence.id = 'practice-before-moving-on';
    sequence.innerHTML = `<h2>Practice before moving on</h2><p class="learning-purpose">Work through these questions before starting the next lesson. Keep the equations and shapes beside your code while you answer.</p><section class="learning-part"><h3>Code walkthrough</h3><p>${esc(task.bridge)}</p><div class="code-walkthrough">${walkthroughHtml}</div></section><section id="code-checkpoint" class="learning-part"><h3>Coding challenge</h3><p>${esc(task.task)}</p><details><summary>Reference implementation</summary><pre><code>${esc(task.solution)}</code></pre><p class="answer-note">Compare the assertions and shapes with your own implementation.</p></details></section><section class="learning-part"><h3>Exercises</h3><ol>${questions.map((item) => `<li>${esc(item)}</li>`).join('')}</ol><details><summary>Answers</summary><ol>${answers.map((item) => `<li>${esc(item)}</li>`).join('')}</ol></details></section>`;
    host.prepend(sequence);
    const cards = [...host.children].filter((node) => node.classList.contains('studio-card'));
    cards.forEach((card) => card.remove());
  });
})();
