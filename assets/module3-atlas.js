/* Each lesson gets a visual grammar that represents its computation.
   Shared primitives provide typography and interaction, not a box template. */
(() => {
  const id = Number(document.body.dataset.lesson);
  const root = document.querySelector('main > .diagram');
  if (!root || id < 15 || id > 22) return;
  const tokens = ['The', 'cat', 'sat', '.'];
  const state = { token: 2, batch: 0, mode: 'rms', split: false, scale: 1, x1: 1, x2: -1 };
  const fmt = n => Number((Math.abs(n) < 0.0005 ? 0 : n).toFixed(3)).toString();
  const text = (x, y, label, cls = '', anchor = 'start') => `<text x="${x}" y="${y}" class="${cls}" text-anchor="${anchor}">${label}</text>`;
  const path = (d, cls = '', extra = '') => `<path d="${d}" class="${cls}" ${extra}/>`;
  const dot = (x, y, r = 7, cls = '') => `<circle cx="${x}" cy="${y}" r="${r}" class="${cls}"/>`;
  const math = value => `\\(${value}\\)`;
  const arrow = 'marker-end="url(#study-arrow)"';
  const curve = (x, y, x2, y2, cls = '', width = 1.5) => path(`M${x} ${y} C${x} ${(y+y2)/2} ${x2} ${(y+y2)/2} ${x2} ${y2}`, cls, `stroke-width="${width}"`);
  const button = (label, key, value, selected) => `<button type="button" data-choice="${key}" data-value="${value}" aria-pressed="${selected}">${label}</button>`;
  const tokenControls = () => `<span>Query token</span>${tokens.map((v,i)=>button(v,'token',i,state.token===i)).join('')}`;
  const range = (label,key,min,max,step) => `<label>${label}<input aria-label="${label}" data-range="${key}" type="range" min="${min}" max="${max}" step="${step}" value="${state[key]}"></label>`;
  const spec = {
    15: ['Find one token inside the tensor.', 'Choose a sequence and a token. Only the feature coordinates remain.', 'X[b,t,:]\\in\\mathbb{R}^{D}', 'embedding'],
    16: ['Move the values. Keep the axes.', 'Follow the same three features before and after normalization.', '\\operatorname{RMSNorm}(x)=\\frac{x}{\\sqrt{\\frac{1}{D}\\sum_j x_j^2+\\epsilon}}\\odot\\gamma', 'norm'],
    17: ['One token. Three learned views.', 'Project the same input into queries, keys, and values. Heads partition features.', 'Q=XW_Q,\\quad K=XW_K,\\quad V=XW_V', 'attention'],
    18: ['The future loses its connections.', 'Move the query position to see its causal field grow. Stored keys stay in place.', 'P_{ij}=0\\quad\\text{for }j>i', 'attention'],
    19: ['Weights choose. Values contribute.', 'Trace each probability into its two-coordinate contribution to the output.', 'o_i=\\sum_{j\\le i}p_{ij}v_j,\\qquad\\sum_jp_{ij}=1', 'attention'],
    20: ['Same sentence. Different attention.', 'Both heads see every token. Their learned comparisons produce different weights.', 'O=\\operatorname{Concat}(O_1,\\ldots,O_H)W_O', 'attention'],
    21: ['A correction moves the state.', 'The original coordinates stay visible as a learned update moves each one.', 'Y=X+\\alpha F,\\qquad X,F,Y\\in\\mathbb{R}^{B\\times T\\times D}', 'block'],
    22: ['A gate shapes each feature.', 'Two learned views meet coordinate by coordinate, then project back to model width.', '\\operatorname{FFN}(x)=(\\operatorname{SiLU}(xW_g)\\odot xW_u)W_d', 'mlp']
  }[id];
  root.className = 'diagram study-atlas';
  root.id = 'lesson-visual';
  root.innerHTML = `<header class="study-heading"><p class="eyebrow">INSIDE THE TRANSFORMER · ${String(id).padStart(2,'0')}</p><h2>${spec[0]}</h2><p>${spec[1]}</p></header><div class="study-controls"></div><div class="study-canvas"></div><div class="study-equation">${math(spec[2])}</div><footer class="study-footer"><p aria-live="polite" data-study-readout></p><a href="#code-checkpoint">Read this lesson's code ↓</a></footer>`;
  const controls = root.querySelector('.study-controls');
  const canvas = root.querySelector('.study-canvas');
  const readout = root.querySelector('[data-study-readout]');

  // Small deterministic examples, shared with the numerical walkthroughs.
  const batches = [
    [[.4,-.2,1.1,0,.7,-.5],[.1,.8,-.3,1.2,.2,.4],[-.6,.3,.9,-.1,1.4,.2],[.2,-.7,.5,.8,-.4,1]],
    [[-.3,.6,.1,1.3,-.2,.8],[.7,.2,-.5,.4,1.1,-.1],[1,-.4,.3,.6,.2,-.8],[-.2,1.2,.4,-.6,.9,.1]]
  ];
  const scores = [.7,1.1,.2,2];
  const values = [[1,0],[0,2],[1,1],[-1,1]];
  const weights = () => {
    const max = Math.max(...scores.slice(0,state.token+1));
    const exp = scores.map((s,j)=>j<=state.token ? Math.exp(s-max) : 0);
    const total = exp.reduce((a,b)=>a+b,0);
    return exp.map(v=>v/total);
  };

  function tensor() {
    let s = text(32,32,'BATCH INDEX '+state.batch,'meta') + text(184,32,'FEATURE COORDINATES →','meta');
    const matrix = batches[state.batch];
    matrix.forEach((row,i)=>{
      const y=90+i*64;
      s += `<g role="button" tabindex="0" data-pick="${i}" aria-label="Select token ${tokens[i]}" aria-pressed="${i===state.token}">`;
      s += `<rect x="28" y="${y-26}" width="440" height="52" class="hit"/>`;
      s += text(44,y+5,tokens[i],i===state.token?'label accent':'label');
      if(i===state.token) s+=path(`M140 ${y+32} H464`,'accent-stroke');
      row.forEach((v,j)=>{
        s+=dot(168+j*52,y,3+Math.abs(v)*7, i===state.token?'solid':'soft-dot');
        s+=text(168+j*52,y+23,fmt(v),'number','middle');
      });
      s+='</g>';
    });
    const y=90+state.token*64;
    s+=path(`M480 ${y} H504 V184 H548`,'wire',arrow);
    s+=text(584,68,'One token, six features.','headline');
    s+=text(584,96,`X[${state.batch}, ${state.token}, :]`,'mono');
    s+=path('M568 216 H932','axis');
    matrix[state.token].forEach((v,j)=>{
      const x=590+j*62, end=216-v*62;
      s+=path(`M${x} 216 V${end}`,'value-stem')+dot(x,end,5,'solid')+text(x,330,`f${j}`,'mono','middle');
    });
    s+=text(568,360,'Length shows value; row selection preserves width.','note');
    return [s,`Zero-based indexing: X[${state.batch}, ${state.token}, :] = (${matrix[state.token].map(fmt).join(', ')}).`];
  }
  function normalize() {
    const input=[1,2,3], rms=Math.sqrt(14/3), sigma=Math.sqrt(2/3);
    const output=input.map(x=>state.mode==='rms'?x/rms:(x-2)/sigma);
    const y=v=>300-v*60;
    let s=text(220,38,'Input','headline','middle')+text(636,38,state.mode==='rms'?'RMSNorm':'LayerNorm','headline','middle');
    [-1,0,1,2,3].forEach(v=>{s+=path(`M172 ${y(v)} H700`,v===0?'axis':'grid')+text(136,y(v)+5,v,'number','end');});
    s+=path('M220 92 V372 M636 92 V372','axis');
    input.forEach((v,i)=>{
      s+=path(`M228 ${y(v)} L628 ${y(output[i])}`,'value-stem')+dot(220,y(v),7)+dot(636,y(output[i]),7,'solid');
      s+=text(204,y(v)-14,`f${i} · ${v}`,'mono','end')+text(656,y(output[i])+5,fmt(output[i]),'mono');
    });
    s+=text(784,126,'Same token.','headline')+text(784,154,'Same width.','headline')+text(784,200,'Only its feature','note')+text(784,222,'values change.','note');
    return [s,state.mode==='rms'?`RMS = ${fmt(rms)}. Divide each coordinate by the same scale; no centering. γ = 1, ε omitted in this toy example.`:`Mean = 2; standard deviation = ${fmt(sigma)}. Center, then rescale. γ = 1, β = 0, ε omitted.`];
  }
  function project() {
    const rows=[[1,2,0,-1],[1,2,-1,0],[2,2,0,1]];
    let s=text(40,174,'One token','headline')+text(40,204,'x = (1, 2, 0, −1)','mono');
    s+=path('M204 190 H256 M256 78 V318','wire');
    rows.forEach((row,i)=>{
      const y=78+i*120;
      s+=path(`M256 ${y} H344`,'wire',arrow)+text(270,y-16,['q_proj','k_proj','v_proj'][i],'mono');
      s+=text(368,y+8,['Q','K','V'][i],'headline');
      row.forEach((v,j)=>{
        const x=480+j*104;
        s+=dot(x,y,18,'token-dot')+text(x,y+5,fmt(v),'label','middle');
      });
      if(state.split) {
        s+=path(`M454 ${y+28} V${y+36} H610 V${y+28} M662 ${y+28} V${y+36} H818 V${y+28}`,'accent-stroke');
        s+=text(532,y+56,'head 1','mono','middle')+text(740,y+56,'head 2','mono','middle');
      }
    });
    s+=text(868,88,'asks','note')+text(868,208,'is compared','note')+text(868,328,'carries content','note');
    return [s,state.split?'Four feature coordinates become two heads × two features. Every head still keeps every token.':'Toy projections produce three different feature views. Switch to heads to reveal the feature partition.'];
  }
  function mask() {
    let s=text(68,40,'KEYS · ALL REMAIN STORED','meta');
    tokens.forEach((token,j)=>{
      const x=112+j*180,allowed=j<=state.token;
      s+=text(x,78,token,'label','middle')+dot(x,108,12,allowed?'token-dot':'soft-dot');
      if(allowed) s+=curve(x,122,380,280,'mix',2);
      else s+=path(`M${x-7} 150 l14 14 M${x+7} 150 l-14 14`,'muted-stroke');
      s+=text(x,188,allowed?'':'future','mono','middle');
    });
    s+=dot(380,296,14,'focal')+text(380,338,`query: ${tokens[state.token]}`,'label','middle');
    s+=text(784,114,`${state.token+1} of 4`,'headline')+text(784,145,'positions are visible.','note')+text(784,204,'Future score: −∞','mono')+text(784,230,'Future weight: 0','mono');
    return [s,`Query position ${state.token+1} can use keys 1–${state.token+1}. No key/value vector is deleted; only future connections are masked.`];
  }
  function blend() {
    const w=weights(), output=[0,0];
    let s=text(40,38,'KEY / SCORE','meta')+text(240,38,'SOFTMAX WEIGHT','meta')+text(468,38,'WEIGHT × VALUE','meta');
    w.forEach((p,j)=>{
      const y=92+j*76;
      output[0]+=p*values[j][0];output[1]+=p*values[j][1];
      s+=text(40,y,tokens[j],'label')+text(96,y,p?fmt(scores[j]):'−∞','mono');
      s+=path(`M240 ${y-6} H420`,'grid')+path(`M240 ${y-6} h${p*180}`,'weight',`stroke-width="8"`)+text(240,y+22,fmt(p),'mono');
      s+=text(468,y,`${fmt(p)} × (${values[j].join(', ')})`,'mono');
      if(p) s+=path(`M660 ${y-6} C716 ${y-6} 716 192 766 192`,'mix',`stroke-width="${1+p*5}"`);
    });
    s+=dot(788,192,20,'focal')+text(788,199,'+','label','middle');
    s+=text(840,182,'Output','headline')+text(840,216,`(${output.map(fmt).join(', ')})`,'mono');
    return [s,`Probabilities sum to 1. Output = (${output.map(fmt).join(', ')}). Stroke thickness shows contribution weight, not value magnitude.`];
  }
  function heads() {
    const matrices=[[[1,0,0,0],[.42,.58,0,0],[.25,.45,.30,0],[.10,.20,.30,.40]],[[1,0,0,0],[.75,.25,0,0],[.15,.20,.65,0],[.05,.10,.25,.60]]];
    let s='';
    matrices.forEach((m,h)=>{
      const offset=40+h*512, center=offset+180;
      s+=text(offset,38,`Head ${h+1}`,'headline');
      tokens.forEach((v,j)=>{
        const x=offset+j*112,p=m[state.token][j];
        s+=text(x,84,v,'label','middle')+dot(x,110,9,'token-dot')+text(x,146,fmt(p),'mono','middle');
        if(p) s+=curve(x,164,center,288,'mix',1+p*7);
      });
      s+=dot(center,302,12,'focal')+text(center,344,`query: ${tokens[state.token]}`,'label','middle');
    });
    return [s,`Both heads use query “${tokens[state.token]}” and retain all four token positions. Line thickness encodes each head’s attention probabilities.`];
  }
  function residual() {
    const x=[1,-2,.5],delta=[.2,.3,-.1].map(v=>v*state.scale);
    const map=v=>220+(v+2.5)*140;
    let s=text(40,36,'FEATURE','meta')+text(220,36,'VALUE →','meta');
    x.forEach((v,i)=>{
      const y=106+i*110, end=v+delta[i];
      s+=text(40,y+5,`f${i}`,'label')+path(`M220 ${y} H920`,'axis');
      [-2,-1,0,1,2].forEach(t=>{s+=path(`M${map(t)} ${y-5} v10`,'axis')+text(map(t),y+26,t,'number','middle');});
      s+=dot(map(v),y,7)+path(`M${map(v)} ${y-22} H${map(end)}`,'accent-stroke',arrow)+dot(map(end),y,5,'focal');
      s+=text(948,y+5,fmt(end),'mono');
      s+=text(map(v),y-42,`Δ = ${fmt(delta[i])}`,'mono','middle');
    });
    return [s,`Hollow dots are X; filled dots are Y. α = ${fmt(state.scale)} gives Y = (${x.map((v,i)=>fmt(v+delta[i])).join(', ')}). At α = 0 the identity path remains.`];
  }
  function swiglu() {
    const x=[state.x1,state.x2], gate=x.map(v=>v/(1+Math.exp(-v))),up=[x[0]+x[1],x[0]-x[1]],z=gate.map((v,i)=>v*up[i]),out=[z[0],-2*z[1]];
    let s=text(40,44,`x = (${x.map(fmt).join(', ')})`,'headline')+text(40,72,'ONE TOKEN · TWO FEATURES','meta');
    s+=text(316,40,'SiLU gate','headline')+text(536,40,'Up branch','headline')+text(772,40,'Product → output','headline');
    // A miniature SiLU curve makes the gate’s signed response visible.
    s+=path('M48 230 H236 M142 120 V266','axis');
    const pts=Array.from({length:61},(_,i)=>{const v=-3+i*.1;return `${i?'L':'M'}${142+v*30} ${230-v/(1+Math.exp(-v))*32}`;}).join(' ');
    s+=path(pts,'mix')+text(48,294,'SiLU keeps a small','note')+text(48,316,'negative response.','note');
    x.forEach((v,i)=>{
      const y=146+i*174;
      s+=dot(142+v*30,230-gate[i]*32,5,'focal');
      s+=text(308,y-30,`f${i}`,'mono');
      s+=path(`M320 ${y} H446 M536 ${y} H662`,'axis');
      s+=path(`M380 ${y} v${-gate[i]*24}`,'value-stem')+dot(380,y-gate[i]*24,5,'solid');
      s+=path(`M596 ${y} v${-up[i]*16}`,'value-stem')+dot(596,y-up[i]*16,5,'solid');
      s+=text(380,y+28,fmt(gate[i]),'mono','middle')+text(596,y+28,fmt(up[i]),'mono','middle');
      s+=path(`M446 ${y} H466 V${y+60} H704 V${y+18} M662 ${y} H686`,'wire');
      s+=dot(704,y,17,'token-dot')+text(704,y+5,'×','label','middle');
      s+=path(`M722 ${y} H764`,'wire',arrow)+text(784,y+5,fmt(z[i]),'mono');
      s+=text(860,y+5,'→','note')+text(924,y+5,fmt(out[i]),'label');
    });
    return [s,`Toy weights: Wg = I; up = (x₁+x₂, x₁−x₂); Wd = diag(1, −2). Output = (${out.map(fmt).join(', ')}). A real MLP usually expands D to a larger intermediate width.`];
  }
  const renderers={15:tensor,16:normalize,17:project,18:mask,19:blend,20:heads,21:residual,22:swiglu};
  function render() {
    const [drawing,caption]=renderers[id]();
    canvas.innerHTML=`<svg viewBox="0 0 1040 410" role="group" aria-label="${spec[0]}" xmlns="http://www.w3.org/2000/svg"><defs><marker id="study-arrow" markerWidth="5" markerHeight="5" refX="5" refY="2.5" orient="auto"><path d="M0 0 L5 2.5 L0 5Z" fill="var(--muted)" stroke="none"/></marker></defs>${drawing}</svg>`;
    readout.textContent=caption;
    canvas.querySelectorAll('[data-pick]').forEach(node=>{
      const select=()=>{state.token=Number(node.dataset.pick);render();canvas.querySelector(`[data-pick="${state.token}"]`).focus({preventScroll:true});};
      node.addEventListener('click',select);
      node.addEventListener('keydown',e=>{if(e.key==='Enter'||e.key===' '){e.preventDefault();select();}});
    });
  }
  if(id===15) controls.innerHTML='<span>Sequence</span>'+[0,1].map(i=>button(`Batch ${i+1}`,'batch',i,i===state.batch)).join('')+'<span class="study-hint">Click a token row below</span>';
  if(id===16) controls.innerHTML=['rms','layer'].map(v=>button(v==='rms'?'RMSNorm':'LayerNorm','mode',v,v===state.mode)).join('');
  if(id===17) controls.innerHTML=button('Projected features','split',false,true)+button('Reveal heads','split',true,false);
  if([18,19,20].includes(id)) controls.innerHTML=tokenControls();
  if(id===21) controls.innerHTML=range('Update strength α','scale',0,1.5,.1);
  if(id===22) controls.innerHTML=range('Input x₁','x1',-2,2,.1)+range('Input x₂','x2',-2,2,.1);
  controls.addEventListener('click',e=>{
    const b=e.target.closest('[data-choice]');if(!b)return;
    const key=b.dataset.choice;
    state[key]=key==='split'?b.dataset.value==='true':key==='mode'?b.dataset.value:Number(b.dataset.value);
    controls.querySelectorAll(`[data-choice="${key}"]`).forEach(n=>n.setAttribute('aria-pressed',String(n===b)));
    render();
    if(key==='mode') {
      const equation=root.querySelector('.study-equation');
      window.MathJax?.typesetClear?.([equation]);
      equation.innerHTML=math(state.mode==='rms'?spec[2]:'\\operatorname{LayerNorm}(x)=\\frac{x-\\mu}{\\sqrt{\\sigma^2+\\epsilon}}\\odot\\gamma+\\beta');
      window.MathJax?.typesetPromise?.([equation]);
    }
  });
  controls.addEventListener('input',e=>{if(e.target.dataset.range){state[e.target.dataset.range]=Number(e.target.value);render();}});
  render();
  // Math remains outside the frequently updated SVG so slider drags do not
  // rebuild typesetting or interrupt assistive-technology focus.
  const typeset=()=>window.MathJax?.typesetPromise?.([root.querySelector('.study-equation')]);
  if(window.MathJax?.startup?.promise) window.MathJax.startup.promise.then(typeset);
  else window.addEventListener('load',typeset,{once:true});
})();
