(() => {
  const f = (value) => value.replaceAll('§', String.fromCharCode(92));
  const maps = {
    '0001': ['How a long vector becomes rotation planes', 'RoPE works pair by pair. The vector keeps its width while every local 2D plane rotates.', ['feature vector', 'adjacent pairs', '2D rotations'], [
      ['Read the vector', 'Start with ordered feature coordinates.', '§(q=[q_0,q_1,q_2,q_3,q_4,q_5]§)', 'features'],
      ['Group adjacent coordinates', 'Each pair becomes one small plane.', '§((q_0,q_1),(q_2,q_3),(q_4,q_5)§)', 'pairs'],
      ['Rotate each plane', 'Position chooses an angle inside every pair.', '§(x_i^{§prime}=R_{m§theta_i}x_i§)', 'circle']]],
    '0002': ['How an angle becomes coordinates', 'Radians connect distance around a circle to cosine and sine, the exact entries used by rotation matrices.', ['radian angle', 'unit-circle point', 'rotation coordinates'], [
      ['Measure the turn', 'Radians compare arc length with radius.', '§(§theta=s/r§)', 'circle'],
      ['Locate the point', 'Cosine is horizontal; sine is vertical.', '§((x,y)=(§cos§theta,§sin§theta)§)', 'plane'],
      ['Advance with position', 'Every token adds one frequency step.', '§(§phi_{m,i}=m§theta_i§)', 'frequency']]],
    '0003': ['How a complex number stores a 2D vector', 'Complex notation packages the same horizontal and vertical coordinates into one number.', ['point (x,y)', 'number x + iy', 'magnitude and phase'], [
      ['Start with a point', 'Two real coordinates locate one vector.', '§(§mathbf v=(x,y)§)', 'plane'],
      ['Package coordinates', 'The imaginary unit labels the vertical coordinate.', '§(z=x+iy§)', 'features'],
      ['Read length and angle', 'Polar form separates magnitude from phase.', '§(z=re^{i§phi}§)', 'circle']]],
    '0004': ['Why complex multiplication rotates', 'Multiplying by a unit complex phase adds an angle while preserving the vector length.', ['vector phase', 'unit phase', 'rotated vector'], [
      ['Write polar form', 'Magnitude and phase are explicit.', '§(z=re^{i§phi}§)', 'circle'],
      ['Multiply by a phase', 'The new phase adds to the old one.', '§(ze^{i§theta}=re^{i(§phi+§theta)}§)', 'attention'],
      ['Keep the magnitude', 'Only direction changes.', '§(|ze^{i§theta}|=|z|§)', 'circle']]],
    '0005': ['How a matrix performs the same rotation', 'The 2×2 rotation matrix is the real-coordinate form of multiplying by a unit complex phase.', ['input coordinates', 'rotation matrix', 'output coordinates'], [
      ['Build the matrix', 'Cosine and sine encode rotated axes.', '§(R_§theta=§begin{bmatrix}§cos§theta&amp;-§sin§theta§§§sin§theta&amp;§cos§theta§end{bmatrix}§)', 'matrix'],
      ['Multiply a vector', 'Each output combines both input coordinates.', '§(§mathbf x^{§prime}=R_§theta§mathbf x§)', 'features'],
      ['Read the motion', 'The direction turns counterclockwise.', '§(R_{§pi/2}(1,0)=(0,1)§)', 'circle']]],
    '0006': ['How transpose and dot products preserve geometry', 'Rotation matrices are orthogonal: transpose reverses the turn and preserves geometric comparisons.', ['dot product', 'transpose', 'preserved comparison'], [
      ['Compare vectors', 'A dot product measures signed alignment.', '§(§mathbf q^{§mathsf T}§mathbf k=§sum_iq_ik_i§)', 'attention'],
      ['Transpose rotation', 'For rotations, transpose equals inverse.', '§(R_§theta^{§mathsf T}=R_{-§theta}§)', 'matrix'],
      ['Preserve length', 'Rotating does not change squared magnitude.', '§((R§mathbf x)^{§mathsf T}(R§mathbf x)=§mathbf x^{§mathsf T}§mathbf x§)', 'circle']]],
    '0007': ['Where RoPE enters attention', 'RoPE changes Q/K geometry before the dot-product comparison; it does not create the weights by itself.', ['Q/K/V projection', 'rotate Q and K', 'query-key score'], [
      ['Project three views', 'Q asks, K is compared, V carries content.', '§(Q=XW_Q,§ K=XW_K,§ V=XW_V§)', 'branch'],
      ['Rotate Q and K', 'Position modifies comparison vectors, not V.', '§(Q^{§prime}=§operatorname{RoPE}(Q),§ K^{§prime}=§operatorname{RoPE}(K)§)', 'circle'],
      ['Form a score', 'The rotated dot product enters attention.', '§(s_{mn}=q_m^{§prime§mathsf T}k_n^{§prime}§)', 'attention']]],
    '0008': ['How position becomes multiple frequencies', 'Fast pairs distinguish nearby positions; slow pairs preserve longer-range position patterns.', ['pair index i', 'frequency θᵢ', 'phase mθᵢ'], [
      ['Assign pair indices', 'Two head features share one frequency.', '§(i=0,1,§ldots,d_h/2-1§)', 'pairs'],
      ['Choose frequencies', 'Later pairs rotate more slowly.', '§(§theta_i=10000^{-2i/d_h}§)', 'frequency'],
      ['Multiply by position', 'Position becomes phase at every scale.', '§(§phi_{m,i}=m§theta_i§)', 'circle']]],
    '0009': ['How relative position appears in the score', 'Absolute query and key rotations combine into one rotation by their displacement \(n-m\).', ['rotate by m and n', 'combine rotations', 'relative n − m'], [
      ['Rotate two positions', 'Query and key receive different phases.', '§(q_m^{§prime}=R_mq_m,§ k_n^{§prime}=R_nk_n§)', 'distance'],
      ['Move rotations together', 'Transpose places both rotations side by side.', '§(q_m^{§prime§mathsf T}k_n^{§prime}=q_m^{§mathsf T}R_m^{§mathsf T}R_nk_n§)', 'matrix'],
      ['Collapse displacement', 'Composition leaves only relative offset.', '§(R_m^{§mathsf T}R_n=R_{n-m}§)', 'distance']]],
    '0010': ['How to read a hidden-state tensor', 'Batch selects a sequence, token selects a row, and feature selects one coordinate of that token representation.', ['batch B', 'token T', 'features d_model'], [
      ['Name every axis', 'Do not treat tensor dimensions as interchangeable.', '§(X§in§mathbb R^{B§times T§times d_{§mathrm{model}}}§)', 'tensor'],
      ['Select one token', 'Fix batch and token; keep every feature.', '§(X[b,t,:]§in§mathbb R^{d_{§mathrm{model}}}§)', 'features'],
      ['Preserve outer shape', 'Residual additions require model width again.', '§(B§times T§times d_{§mathrm{model}}§to B§times T§times d_{§mathrm{model}}§)', 'sum']]],
    '0011': ['How normalization changes values, not axes', 'Statistics come from one token’s features. Token positions remain separate and tensor shape stays fixed.', ['token vector', 'feature statistics', 'scaled vector'], [
      ['Take one token row', 'Normalization acts independently per position.', '§(§mathbf x=(x_1,§ldots,x_d)§)', 'features'],
      ['Compute a scale', 'LayerNorm centers; RMSNorm only rescales.', '§(§mu,§sigma^2§quad§text{or}§quad§operatorname{RMS}(§mathbf x)§)', 'bars'],
      ['Return same width', 'Only feature values change.', '§(§operatorname{Norm}:§mathbb R^d§to§mathbb R^d§)', 'features']]],
    '0012': ['How one token branches into Q, K, and V', 'Projection changes feature meaning. Splitting heads reorganizes features but never divides the sentence.', ['normalized token', 'three projections', 'head feature chunks'], [
      ['Start from normalized X', 'The same row feeds all three matrices.', '§(§bar X§in§mathbb R^{B§times T§times d_{§mathrm{model}}}§)', 'tensor'],
      ['Project three views', 'Q asks, K is compared, V carries content.', '§(Q=§bar XW_Q,§ K=§bar XW_K,§ V=§bar XW_V§)', 'branch'],
      ['Split feature width', 'Model width becomes heads times head width.', '§(d_{§mathrm{model}}=h§,d_h§)', 'pairs']]],
    '0013': ['How a causal mask removes future connections', 'The mask acts on connections, not stored vectors. It precedes softmax so forbidden weights become zero.', ['Q′K′ scores', 'triangular mask', 'future weight = 0'], [
      ['Build all scores', 'Each query row compares with every key.', '§(S=Q^{§prime}K^{§prime§mathsf T}/§sqrt{d_h}§)', 'attention'],
      ['Mark the future', 'Columns with key index \(j>i\) are forbidden.', '§(M_{ij}=-§infty§quad§text{when }j>i§)', 'mask'],
      ['Normalize safely', 'Exponentiating negative infinity yields zero.', '§(P=§operatorname{softmax}(S+M)§)', 'bars']]],
    '0014': ['How scores become a weighted content blend', 'Q/K comparisons decide how much each position contributes; V supplies the content being combined.', ['masked scores', 'softmax weights', 'weighted V sum'], [
      ['Exponentiate scores', 'Larger allowed scores receive more mass.', '§(e^{A_{ij}}§)', 'bars'],
      ['Normalize the row', 'Allowed weights are nonnegative and total one.', '§(P_{ij}=e^{A_{ij}}/§sum_re^{A_{ir}}§)', 'bars'],
      ['Blend values', 'One output combines allowed value vectors.', '§(§mathbf o_i=§sum_jP_{ij}§mathbf v_j§)', 'sum']]],
    '0015': ['How multiple heads share one sequence', 'Heads specialize through different feature projections while all of them compare the same token sequence.', ['full sequence', 'parallel head spaces', 'joined output'], [
      ['Split feature width', 'Every token supplies \(d_h\) features per head.', '§(B§times T§times d_{§mathrm{model}}§to B§times h§times T§times d_h§)', 'pairs'],
      ['Attend in parallel', 'Each head builds its own token map.', '§(P_1,§ldots,P_h§in§mathbb R^{T§times T}§)', 'heads'],
      ['Join features', 'Concatenation and \(W_O\) restore width.', '§(O_{§mathrm{attn}}=§operatorname{Concat}(O_1,§ldots,O_h)W_O§)', 'branch']]],
    '0016': ['How a residual path carries and updates state', 'The identity route remains available while the sublayer proposes a correction of exactly the same shape.', ['current state X', 'learned update F(X)', 'new state X + F(X)'], [
      ['Keep identity', 'The hidden state travels directly to addition.', '§(X§)', 'features'],
      ['Compute correction', 'Attention or FFN produces a learned update.', '§(F(§operatorname{Norm}(X))§)', 'branch'],
      ['Add coordinates', 'Matching shapes produce the next state.', '§(Y=X+F(§operatorname{Norm}(X))§)', 'sum']]],
    '0017': ['How two SwiGLU branches gate features', 'One expanded branch sets a signed gate; the other carries candidate values. Their product is projected back.', ['gate and up', 'coordinate product', 'down projection'], [
      ['Expand twice', 'Two matrices create matching hidden widths.', '§(g=xW_g,§quad u=xW_u§)', 'gate'],
      ['Apply gate', 'SiLU shapes one branch before multiplication.', '§(z=§operatorname{SiLU}(g)§odot u§)', 'gate'],
      ['Return to model width', 'Down projection enables residual addition.', '§(§operatorname{FFN}(x)=zW_d§)', 'branch']]],
    '0018': ['How all block stages connect', 'Position affects Q/K comparison, attention mixes tokens, SwiGLU transforms features, and residual paths carry state.', ['normalize and attend', 'first residual', 'normalize and SwiGLU', 'second residual'], [
      ['Attention sublayer', 'Normalize, project, rotate, mask, and blend.', '§(O_{§mathrm{attn}}=§operatorname{MHA}(§operatorname{Norm}(X))§)', 'block'],
      ['First state update', 'Add attention information to current state.', '§(Y=X+O_{§mathrm{attn}}§)', 'sum'],
      ['Feature update', 'Normalize Y, apply SwiGLU, and add again.', '§(X^{(§ell+1)}=Y+§operatorname{FFN}(§operatorname{Norm}(Y))§)', 'block']]]
  };

  const kinds = {
    features: '<div class="map-features"><span></span><span></span><span></span><span></span><span></span><span></span></div>',
    pairs: '<div class="map-pairs"><b><span></span><span></span></b><b><span></span><span></span></b><b><span></span><span></span></b></div>',
    circle: '<div class="map-circle"></div>',
    plane: '<div class="map-plane"><span></span></div>',
    matrix: '<div class="map-matrix"><div class="map-matrix-grid"><span>cos</span><span>−sin</span><span>sin</span><span>cos</span></div><b>×</b><div class="map-vector"><span>x</span><span>y</span></div></div>',
    attention: '<div class="map-attention"><span class="map-node">Q</span><span class="map-link"></span><span class="map-node key">K</span></div>',
    frequency: '<div class="map-frequency"><span style="--period:25%"></span><span style="--period:45%"></span><span style="--period:68%"></span><span style="--period:92%"></span></div>',
    distance: '<div class="map-distance"><b>m</b><span>relative distance n − m</span><b>n</b></div>',
    tensor: '<div class="map-tensor">' + '<span></span>'.repeat(18) + '</div>',
    branch: '<div class="map-branch"><b>x</b><span style="--branch-color:var(--blue)">Q</span><span>K</span><span>V</span></div>',
    mask: '<div class="map-mask"><span></span><span class="off"></span><span class="off"></span><span class="off"></span><span></span><span></span><span class="off"></span><span class="off"></span><span></span><span></span><span></span><span class="off"></span><span></span><span></span><span></span><span></span></div>',
    bars: '<div class="map-bars"><span style="--bar:46%"></span><span style="--bar:76%"></span><span style="--bar:30%"></span><span style="--bar:0%"></span></div>',
    heads: '<div class="map-heads"><div style="--head:var(--blue)"><span></span><span></span><span></span></div><div><span></span><span></span><span></span></div><div><span></span><span></span><span></span></div></div>',
    sum: '<div class="map-sum"><span>X</span><b>+</b><span>F</span><b>=</b><span>Y</span></div>',
    gate: '<div class="map-gate"><span>SiLU gate</span><span>up branch</span><strong>⊙ coordinate-wise</strong></div>',
    block: '<div class="map-block"><span style="--step-color:var(--blue)">Norm</span><span>QKV</span><span>RoPE</span><span>Mask</span><span>Blend</span><span>Add</span><span>SwiGLU</span><span>Add</span></div>'
  };

  const id = document.body.dataset.lesson;
  const map = maps[id];
  const goal = document.querySelector('.goal');
  if (!map || !goal) return;
  const colors = ['var(--blue)', 'var(--teal)', 'var(--coral)'];
  const section = document.createElement('section');
  section.className = 'lesson-map';
  section.setAttribute('aria-label', 'Visual map for this lesson');
  section.innerHTML = '<div class="lesson-map-header"><h2>' + map[0] + '</h2><span>Visual map</span></div><div class="lesson-map-grid"><div class="lesson-map-stages">' +
    map[3].map((stage, index) => '<article class="lesson-map-stage" style="--map-color:' + colors[index] + '"><span class="lesson-map-number">' + (index + 1) +
      '</span><div class="lesson-map-copy"><h3>' + stage[0] + '</h3><p>' + stage[1] + '</p><div class="lesson-map-formula">' + f(stage[2]) +
      '</div></div><div class="map-schematic">' + kinds[stage[3]] + '</div></article>').join('') +
    '</div><aside class="lesson-map-aside"><strong>Why it connects</strong><p>' + map[1] + '</p><div class="lesson-map-chain">' +
    map[2].map((item, index) => '<span>' + item + '</span>' + (index < map[2].length - 1 ? '<i>↓</i>' : '')).join('') +
    '</div></aside></div>';
  goal.insertAdjacentElement('afterend', section);
})();
