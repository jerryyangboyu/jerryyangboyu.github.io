document
  .querySelectorAll('pre code[class*="language-"]:not([data-source-code])')
  .forEach((code) => {
    if (window.hljs) window.hljs.highlightElement(code);
  });
