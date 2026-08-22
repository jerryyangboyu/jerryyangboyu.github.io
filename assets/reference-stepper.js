(() => {
  document.querySelectorAll('[data-reference-sequence]').forEach((sequence) => {
    const steps = [...sequence.querySelectorAll('[data-reference-step]')];
    const tabs = [...sequence.querySelectorAll('[data-step-target]')];
    const previous = sequence.querySelector('[data-step-prev]');
    const next = sequence.querySelector('[data-step-next]');
    const showAllButton = sequence.querySelector('[data-step-all]');
    const status = sequence.querySelector('[data-step-status]');
    const reducedMotion = window.matchMedia('(prefers-reduced-motion: reduce)');
    let current = Math.max(0, steps.findIndex((step) => `#${step.id}` === window.location.hash));
    let showAll = false;

    tabs.forEach((tab, index) => {
      const tabId = `${steps[index].id}-tab`;
      tab.id = tabId;
      tab.setAttribute('role', 'tab');
      tab.setAttribute('aria-controls', steps[index].id);
      steps[index].setAttribute('role', 'tabpanel');
      steps[index].setAttribute('aria-labelledby', tabId);
    });

    const moveToCurrent = () => {
      steps[current].scrollIntoView({
        behavior: reducedMotion.matches ? 'auto' : 'smooth',
        block: 'start',
      });
    };

    const render = ({ move = false, updateHash = false } = {}) => {
      steps.forEach((step, index) => {
        step.hidden = !showAll && index !== current;
      });

      tabs.forEach((tab, index) => {
        const active = !showAll && index === current;
        tab.setAttribute('aria-selected', String(active));
        if (active) tab.setAttribute('aria-current', 'step');
        else tab.removeAttribute('aria-current');
      });

      previous.disabled = showAll || current === 0;
      next.disabled = showAll || current === steps.length - 1;
      status.textContent = showAll ? `All ${steps.length} steps` : `Step ${current + 1} of ${steps.length}`;
      showAllButton.textContent = showAll ? 'Return to steps' : 'Show all';
      showAllButton.setAttribute('aria-pressed', String(showAll));

      if (updateHash && !showAll) history.replaceState(null, '', `#${steps[current].id}`);
      if (move) moveToCurrent();
    };

    tabs.forEach((tab, index) => {
      tab.addEventListener('click', () => {
        current = index;
        showAll = false;
        render({ move: true, updateHash: true });
      });
    });

    previous.addEventListener('click', () => {
      current = Math.max(0, current - 1);
      render({ move: true, updateHash: true });
    });

    next.addEventListener('click', () => {
      current = Math.min(steps.length - 1, current + 1);
      render({ move: true, updateHash: true });
    });

    showAllButton.addEventListener('click', () => {
      showAll = !showAll;
      render({ move: !showAll });
    });

    render();
  });
})();
