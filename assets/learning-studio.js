(() => {
  const lessonId = document.body?.dataset.lesson;
  const lessons = {
    '0015': {
      title: 'Read a hidden state as a table of token vectors',
      exerciseContext: 'A language model stores a batch of text as a table \\(X\\in\\mathbb{R}^{B\\times T\\times D}\\): \\(B\\) examples, \\(T\\) word positions, and \\(D\\) numbers describing each position.',
      retrieval: 'A model receives \\(X\\in\\mathbb{R}^{B\\times T\\times D}\\). What does one row \\(X[b,t,:]\\) describe?',
      answer: 'Selecting \\(b\\) and \\(t\\) points to one word in one example, so \\(X[b,t,:]\\in\\mathbb{R}^{D}\\). The feature values stay together because they describe the same word position.',
      bridge: 'The embedding step turns each token ID into a \\(D\\)-number description while preserving the batch and word-position grid. Every later operation may rearrange features, but the block must return to \\(X\\in\\mathbb{R}^{B\\times T\\times D}\\) so the next layer can read the same table.',
      code: `token_ids = torch.tensor([[4, 7, 2], [9, 1, 5]])  # [B=2, T=3]\nx = nn.Embedding(32, 8)(token_ids)                  # [B, T, D]\nassert x.shape == (2, 3, 8)`,
      task: 'Write a complete select_token(x, batch, token) function for \\(x\\in\\mathbb{R}^{B\\times T\\times D}\\). Return the one-word vector in \\(\\mathbb{R}^{D}\\), and raise IndexError when the requested example or position does not exist.',
      solution: `def select_token(x, batch, token):\n    B, T, D = x.shape\n    if not (0 <= batch < B and 0 <= token < T):\n        raise IndexError("batch/token outside the tensor")\n    return x[batch, token, :]`,
      practice: ['For \\(B=2,T=3,D=4\\), how many numbers are in \\(X\\)?', 'What shape is \\(X[1,2,:]\\)? What does that row represent?', 'The state has shape \\(\\mathbb{R}^{B\\times T\\times4}\\), but an update has shape \\(\\mathbb{R}^{B\\times T\\times6}\\). Can you add them? Why?'],
      solutions: ['\\(2\\cdot3\\cdot4=24\\).', '\\(\\mathbb{R}^{4}\\); it is the feature vector for position 2 in example 1.', 'No. Elementwise addition needs matching shapes, including the feature width.']
    },
    '0016': {
      title: 'Normalize features independently for each token',
      exerciseContext: 'Before the model makes a decision about a word, it first puts that word’s feature values on a comparable scale. For \\(x\\in\\mathbb{R}^{B\\times T\\times D}\\), RMSNorm works on the \\(D\\) features of each word independently.',
      retrieval: 'For one word row, which values does RMSNorm use to set its scale?',
      answer: 'It combines the \\(D\\) features belonging to one \\((b,t)\\) pair, then keeps the table shape \\(\\mathbb{R}^{B\\times T\\times D}\\). It never lets one word change another word’s scale.',
      bridge: 'MiniMind normalizes the current word descriptions before attention and before the MLP. The operation is a local calibration step: it changes the scale of each row while preserving the \\(B\\times T\\) grid that carries the text.',
      code: `class RMSNorm(nn.Module):\n    def forward(self, x):\n        rms = torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True) + eps)\n        return self.weight * x * rms       # [B, T, D]`,
      task: 'Write a complete rms_norm(x, weight, eps) function. For \\(x\\in\\mathbb{R}^{2\\times3\\times8}\\), normalize the eight features of each word independently and assert that the result still has shape \\(2\\times3\\times8\\).',
      solution: `def rms_norm(x, weight, eps):\n    variance = x.float().pow(2).mean(dim=-1, keepdim=True)\n    scaled = x * torch.rsqrt(variance + eps).to(x.dtype)\n    output = weight * scaled\n    assert output.shape == x.shape\n    return output`,
      practice: ['For \\(x=(1,2,3)\\), compute \\(\\operatorname{RMS}(x)\\).', 'Why must RMSNorm keep the \\(B\\) and \\(T\\) axes separate?', 'If \\(x\\) has shape \\(2\\times5\\times8\\), what shape does the mean have when it uses the last axis with \\(\\text{keepdim}=True\\)?'],
      solutions: ['\\(\\sqrt{(1^2+2^2+3^2)/3}=\\sqrt{14/3}\\approx2.160\\).', 'Each word is scaled from its own features. Mixing \\(T\\) would let one word change another word’s scale.', '\\(2\\times5\\times1\\). The singleton axis broadcasts back across the eight features.']
    },
    '0017': {
      title: 'Split feature width into heads without splitting the sentence',
      exerciseContext: 'Attention gives several small “readers” a view of the same sentence. Starting from \\(X\\in\\mathbb{R}^{B\\times T\\times D}\\), the feature width is divided into \\(H\\) heads of \\(D_h=D/H\\); every head still sees all \\(T\\) word positions.',
      retrieval: 'Attention uses several heads. What does one head keep, and what does it split?',
      answer: 'One head receives all five positions, with \\(D_h=12/3=4\\) features per position. The split creates different feature views; it does not split the sentence into shorter pieces.',
      bridge: 'The projections create different question, lookup, and content views of the same word table. The shape story is \\(\\mathbb{R}^{B\\times T\\times D}\\to\\mathbb{R}^{B\\times H\\times T\\times D_h}\\), then back to \\(\\mathbb{R}^{B\\times T\\times D}\\) after the heads are joined.',
      code: `q = self.q_proj(x).view(B, T, H, Dh).transpose(1, 2)\nk = self.k_proj(x).view(B, T, H, Dh).transpose(1, 2)\nv = self.v_proj(x).view(B, T, H, Dh).transpose(1, 2)\nq, k = apply_rotary_pos_emb(q, k)`,
      task: 'Write a complete split_heads(projected, n_heads) function. Turn \\(\\mathbb{R}^{B\\times T\\times(H\\cdot D_h)}\\) into \\(\\mathbb{R}^{B\\times H\\times T\\times D_h}\\), and assert the new shape.',
      solution: `def split_heads(projected, n_heads):\n    B, T, width = projected.shape\n    assert width % n_heads == 0\n    Dh = width // n_heads\n    heads = projected.view(B, T, n_heads, Dh).transpose(1, 2)\n    assert heads.shape == (B, n_heads, T, Dh)\n    return heads`,
      practice: ['With \\(D=12\\) and \\(H=3\\), what is \\(D_h\\)?', 'Why does every head still receive all \\(T\\) word positions?', 'If \\(D=10\\) and \\(H=3\\), what should configuration validation reject?'],
      solutions: ['\\(D_h=D/H=4\\).', 'The split uses the feature axis \\(D\\), not the word-position axis \\(T\\).', 'Reject it because \\(10\\) cannot be divided evenly into \\(3\\) heads.']
    },
    '0018': {
      title: 'Mask future positions before probability is formed',
      exerciseContext: 'A decoder writes the next word using the words already available. The score table \\(S\\in\\mathbb{R}^{T\\times T}\\) must hide every future position before scores become probabilities.',
      retrieval: 'A decoder is choosing the word at position \\(i\\). Which positions may it read?',
      answer: 'Only \\(j\\le i\\) is allowed. A word can use itself and the past, but never a later word that has not been generated yet.',
      bridge: 'The causal mask is a product rule: before the model turns scores into probabilities, replace every future score with \\(-\\infty\\). Since \\(e^{-\\infty}=0\\), the final row gives weight only to available context.',
      code: `scores = (q @ k.transpose(-2, -1)) / math.sqrt(Dh)\nscores = scores.masked_fill(~causal_mask, torch.finfo(scores.dtype).min)\nweights = torch.softmax(scores.float(), dim=-1)`,
      task: 'Write a complete causal_softmax(scores) function for \\(S\\in\\mathbb{R}^{T\\times T}\\). Hide the entries \\(S_{ij}\\) with \\(j>i\\) before softmax, then test that those probabilities are zero.',
      solution: `def causal_softmax(scores):\n    T = scores.shape[-1]\n    mask = torch.tril(torch.ones(T, T, dtype=torch.bool, device=scores.device))\n    masked = scores.masked_fill(~mask, torch.finfo(scores.dtype).min)\n    probs = torch.softmax(masked.float(), dim=-1).to(scores.dtype)\n    assert torch.allclose(probs.triu(1), torch.zeros_like(probs.triu(1)))\n    return probs`,
      practice: ['For \\(i=3\\), list every legal key position \\(j\\) using zero-based indexing.', 'Why is \\(-\\infty\\) added before softmax, rather than after it?', 'What should the probabilities in one legal row add up to?'],
      solutions: ['\\(j=0,1,2,3\\).', 'Because \\(e^{-\\infty}=0\\). The future positions then receive zero probability during normalization.', 'They should add up to \\(1\\).']
    },
    '0019': {
      title: 'Turn comparisons into a weighted content read',
      exerciseContext: 'After the model decides how much each available word matters, it blends their content into one context vector. Each probability row \\(P_{i,:}\\) is a spending plan over the allowed value vectors.',
      retrieval: 'One row of \\(P\\) tells a word how much to use each available value. What must be true of that row?',
      answer: 'Every row sums to one and contains no negative weights. It is a distribution over the allowed positions, so the output is a weighted average of their value vectors.',
      bridge: 'The comparison table becomes a set of mixing instructions \\(P\\in\\mathbb{R}^{B\\times H\\times T_q\\times T_k}\\). Multiplying \\(P\\) by \\(V\\in\\mathbb{R}^{B\\times H\\times T_k\\times D_h}\\) produces context \\(O\\in\\mathbb{R}^{B\\times H\\times T_q\\times D_h}\\); the key axis is consumed.',
      code: `weights = F.softmax(scores.float(), dim=-1).to(q.dtype)\nassert torch.allclose(weights.float().sum(dim=-1), torch.ones_like(weights[..., 0].float()))\ncontext = weights @ v                       # [B,H,Tq,Dh]`,
      task: 'Write a complete attention_values(scores, values) function. Turn each score row into a stable probability distribution, check \\(\\sum_j P_{ij}=1\\), and return the weighted content vectors.',
      solution: `def attention_values(scores, values):\n    weights = torch.softmax(scores.float(), dim=-1).to(values.dtype)\n    row_sums = weights.float().sum(dim=-1)\n    assert torch.allclose(row_sums, torch.ones_like(row_sums), atol=1e-5)\n    return weights @ values`,
      practice: ['Compute \\(\\operatorname{softmax}(0,0)\\).', 'With weights \\((0.25,0.75)\\) and values \\((2,0)\\), \\((0,4)\\), what context vector is produced?', 'Why does \\(P V\\) remove the key-position axis?'],
      solutions: ['\\((0.5,0.5)\\).', '\\(0.25(2,0)+0.75(0,4)=(0.5,3)\\).', 'The weights sum over the \\(T_k\\) value positions, leaving one vector for each query position.']
    },
    '0020': {
      title: 'Compress K/V heads while preserving Q’s comparisons',
      exerciseContext: 'During generation, the model must remember earlier keys and values. Grouped-query attention keeps many question heads while sharing a smaller set of K/V memory, reducing the cache without changing the sentence length.',
      retrieval: 'GQA has \\(H\\) query heads and \\(H_{\\mathrm{kv}}\\) memory heads. With \\(H=8\\) and \\(H_{\\mathrm{kv}}=2\\), how much sharing happens?',
      answer: 'Each K/V head is reused \\(H/H_{\\mathrm{kv}}=4\\) times. The model keeps eight ways to ask a question while storing only two copies of the key/value memory.',
      bridge: 'GQA is a memory-saving choice for serving a model: \\(H_{\\mathrm{kv}}<H\\) reduces the cached K/V tensor, while repeating each K/V head restores the comparison shape \\(\\mathbb{R}^{B\\times H\\times T\\times D_h}\\).',
      code: `def repeat_kv(x, n_rep):\n    # x: [B, Hkv, T, Dh]\n    return x[:, :, None, :, :].expand(B, Hkv, n_rep, T, Dh)\n        .reshape(B, Hkv * n_rep, T, Dh)`,
      task: 'Write a complete repeat_kv(kv, n_query_heads) function. Starting with \\(kv\\in\\mathbb{R}^{B\\times H_{\\mathrm{kv}}\\times T\\times D_h}\\), reuse each K/V head so the result has \\(H\\) query heads while \\(T\\) and \\(D_h\\) stay unchanged.',
      solution: `def repeat_kv(kv, n_query_heads):\n    B, Hkv, T, Dh = kv.shape\n    assert n_query_heads % Hkv == 0\n    repeats = n_query_heads // Hkv\n    expanded = kv[:, :, None].expand(B, Hkv, repeats, T, Dh)\n    return expanded.reshape(B, n_query_heads, T, Dh)`,
      practice: ['How many query heads reuse each K/V head when \\(H=8\\) and \\(H_{\\mathrm{kv}}=2\\)?', 'Why does sharing K/V heads reduce decoding memory?', 'What must be true of \\(H\\) and \\(H_{\\mathrm{kv}}\\) before repetition is possible?'],
      solutions: ['Each K/V head is reused \\(8/2=4\\) times.', 'The cache stores two K/V heads instead of eight, while the query heads remain separate.', '\\(H\\) must be divisible by \\(H_{\\mathrm{kv}}\\).']
    },
    '0021': {
      title: 'Treat the residual stream as the state contract',
      exerciseContext: 'The residual stream is the model’s running draft of the meaning of the text. Attention and the MLP each propose a correction, but the shared state keeps the same shape \\(X\\in\\mathbb{R}^{B\\times T\\times D}\\) from one update to the next.',
      retrieval: 'The residual stream carries \\(X\\in\\mathbb{R}^{B\\times T\\times D}\\). Self-attention performs token mixing; the MLP performs feature mixing. Why must both updates return to the same \\(B\\times T\\times D\\) shape before the residual stream can continue?',
      answer: 'Token mixing may change one position using other positions, and feature mixing may use a wider hidden dimension. The shape contract keeps every batch item, token position, and model feature aligned for elementwise residual addition.',
      bridge: 'Think of the identity path as a stable document being revised. Attention writes a context correction, producing \\(Y=X+O_{\\mathrm{attn}}\\); the MLP writes a feature correction, producing \\(X^{(\\ell+1)}=Y+F\\). Both edits return to the same \\(B\\times T\\times D\\) contract.',
      code: `residual = hidden_states\nhidden_states = self.self_attn(self.input_layernorm(hidden_states), ...)\nhidden_states = hidden_states + residual\nhidden_states = hidden_states + self.mlp(self.post_attention_layernorm(hidden_states))`,
      task: 'Write a complete residual_block(x, attention_update, mlp_update) function. Check that \\(x\\), both corrections, and the final state share shape \\(\\mathbb{R}^{B\\times T\\times D}\\) before adding them.',
      solution: `def residual_block(x, attention_update, mlp_update):\n    assert attention_update.shape == x.shape\n    state = x + attention_update\n    assert mlp_update.shape == state.shape\n    return state + mlp_update`,
      practice: ['Self-attention lets “sat” read “cat”. Which axis is token mixing, and which dimensions must still match at the residual merge?', 'SwiGLU expands \\(D\\) to \\(D_{\\mathrm{ff}}\\). What does the wider space buy the MLP, and why must the down projection restore \\(D\\) before the residual merge?', 'If the attention update is zero, what does \\(Y=X+O_{\\mathrm{attn}}\\) say about the identity path and the residual stream?'],
      solutions: ['Token mixing acts across \\(T\\). The batch axis, token axis, and final model width must match for the residual add.', 'The wider space lets the MLP build richer feature combinations. The down projection restores \\(D\\) so the update matches the residual stream.', 'If \\(O_{\\mathrm{attn}}=0\\), then \\(Y=X\\). The identity path carries the state forward unchanged through that sublayer.']
    },
    '0022': {
      title: 'Make the SwiGLU width changes explicit',
      exerciseContext: 'The feed-forward sublayer is a feature workshop for each word. It temporarily expands \\(D\\) to \\(D_{\\mathrm{ff}}\\), gates useful features, and then returns the correction to model width \\(D\\).',
      retrieval: 'SwiGLU has a gate branch and an up branch. Why must their widths match?',
      answer: 'The two branches describe the same word in the same expanded coordinate system, so their elementwise product is defined in \\(\\mathbb{R}^{D_{\\mathrm{ff}}}\\). The down projection then returns the correction to \\(\\mathbb{R}^{D}\\).',
      bridge: 'SwiGLU does not decide which words interact. It improves each word’s feature description independently: expand to \\(D_{\\mathrm{ff}}\\), use one branch to control another, then project back so the residual update fits \\(\\mathbb{R}^{B\\times T\\times D}\\).',
      code: `gate = F.silu(self.gate_proj(x))  # [B,T,Dff]\nup = self.up_proj(x)              # [B,T,Dff]\ngated = gate * up\nreturn self.down_proj(gated)      # [B,T,D]`,
      task: 'Write a complete swiglu(x, gate_proj, up_proj, down_proj) function. For \\(x\\in\\mathbb{R}^{B\\times T\\times D}\\), assert that both branches reach \\(D_{\\mathrm{ff}}\\) before their elementwise product and that the returned correction is back in \\(D\\).',
      solution: `def swiglu(x, gate_proj, up_proj, down_proj):\n    gate = F.silu(gate_proj(x))\n    up = up_proj(x)\n    assert gate.shape == up.shape\n    return down_proj(gate * up)`,
      practice: ['Compute \\(\\sigma(0)\\) and \\(\\operatorname{SiLU}(0)\\).', 'After SiLU, the gate is \\((0.5,-1)\\) and the up branch is \\((4,3)\\). What is their elementwise product?', 'For input shape \\(\\mathbb{R}^{B\\times T\\times D}\\), what shape must the down projection return before the residual add?'],
      solutions: ['\\(\\sigma(0)=0.5\\), so \\(\\operatorname{SiLU}(0)=0\\).', '\\((2,-3)\\).', '\\(\\mathbb{R}^{B\\times T\\times D}\\).']
    },
    '0023': {
      title: 'Teach one complete MiniMindBlock without skipping a contract',
      exerciseContext: 'Treat one MiniMind block as one complete service step: read the current text state, let attention add useful context, let the MLP refine each word, and return both the updated state and the memory needed for the next generated word.',
      retrieval: 'A complete block makes two updates and returns memory. What are the two updates?',
      answer: 'Attention writes \\(O_{\\mathrm{attn}}\\), which lets a word use earlier context; the MLP writes \\(F\\), which refines that word’s features. The block returns \\(X^{(\\ell+1)}\\in\\mathbb{R}^{B\\times T_q\\times D}\\) plus cached \\(K,V\\) for the positions already processed.',
      bridge: 'Follow the block as one state transition. The input table \\(X\\) is normalized, attention writes a context correction, the first residual keeps that correction beside the old state, the MLP refines each row, and the second residual returns the updated table. During decoding, the K/V cache remembers earlier words so a new query of length \\(T_q=1\\) does not recompute the whole prefix.',
      code: `def forward(\n    self,\n    hidden_states,\n    position_embeddings=None,\n    past_key_value=None,\n    use_cache=False,\n    attention_mask=None,\n):\n    residual = hidden_states\n    normalized = self.input_layernorm(hidden_states)\n    hidden_states, present_key_value = self.self_attn(\n        normalized,\n        position_embeddings,\n        past_key_value,\n        use_cache,\n        attention_mask,\n    )\n    assert hidden_states.shape == residual.shape\n    hidden_states = hidden_states + residual\n    normalized = self.post_attention_layernorm(hidden_states)\n    mlp_update = self.mlp(normalized)\n    assert mlp_update.shape == residual.shape\n    hidden_states = hidden_states + mlp_update\n    return hidden_states, present_key_value`,
      task: 'Implement a complete MiniMindBlock module. Define its two RMSNorm layers, attention, and MLP; then write forward so \\(X\\to Y\\to X^{(\\ell+1)}\\), every residual correction has shape \\(\\mathbb{R}^{B\\times T_q\\times D}\\), and a cached decode call supports \\(T_q=1\\).',
      solution: `class MiniMindBlock(nn.Module):\n    def __init__(self, config):\n        super().__init__()\n        self.input_layernorm = RMSNorm(config.hidden_size, config.rms_norm_eps)\n        self.self_attn = MiniMindAttention(config)\n        self.post_attention_layernorm = RMSNorm(config.hidden_size, config.rms_norm_eps)\n        self.mlp = FeedForward(config)\n\n    def forward(\n        self, hidden_states, position_embeddings=None,\n        past_key_value=None, use_cache=False, attention_mask=None\n    ):\n        assert hidden_states.ndim == 3\n        residual = hidden_states\n        hidden_states, present = self.self_attn(\n            self.input_layernorm(hidden_states),\n            position_embeddings, past_key_value,\n            use_cache, attention_mask\n        )\n        assert hidden_states.shape == residual.shape\n        hidden_states = hidden_states + residual\n        mlp_out = self.mlp(self.post_attention_layernorm(hidden_states))\n        assert mlp_out.shape == residual.shape\n        hidden_states = hidden_states + mlp_out\n        return hidden_states, present`,
      practice: ['In \\(Y=X+O_{\\mathrm{attn}}\\), where can a word get information from?', 'In \\(F=\\operatorname{SwiGLU}(U)\\), what changes without moving information between word positions?', 'During cached decoding, which length grows: the new query length \\(T_q\\), or the stored key/value length \\(T_k\\)?'],
      solutions: ['From the current word and the earlier allowed words through causal attention.', 'The feature values inside each word are refined; the word positions stay separate.', 'The stored key/value length \\(T_k\\) grows. A new decode step can keep \\(T_q=1\\).']
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
    sequence.innerHTML = `<h2>Practice before moving on</h2><section class="learning-part"><h3>Code walkthrough</h3><p>${esc(task.bridge)}</p><div class="code-walkthrough">${walkthroughHtml}</div></section><section id="code-checkpoint" class="learning-part"><h3>Coding challenge</h3><p>${esc(task.task)}</p><details><summary>Reference implementation</summary><pre><code class="language-python">${esc(task.solution)}</code></pre><p class="answer-note">Compare the assertions and shapes with your own implementation.</p></details></section><section class="learning-part"><h3>Exercises</h3><p class="exercise-context">${esc(task.exerciseContext || '')}</p><ol>${questions.map((item) => `<li>${esc(item)}</li>`).join('')}</ol><details><summary>Answers</summary><ol>${answers.map((item) => `<li>${esc(item)}</li>`).join('')}</ol></details></section>`;
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
