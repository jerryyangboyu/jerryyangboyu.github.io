const sourceTarget = document.querySelector("[data-source-code]");

if (sourceTarget) {
  fetch("transformer_model.py")
    .then((response) => {
      if (!response.ok) throw new Error(`HTTP ${response.status}`);
      return response.text();
    })
    .then((source) => {
      sourceTarget.textContent = source;
      if (window.hljs) window.hljs.highlightElement(sourceTarget);
    })
    .catch(() => {
      sourceTarget.textContent = "The inline preview could not load. Use the download link above to open transformer_model.py.";
    });
}
