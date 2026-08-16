(() => {
  const lessonId = document.body.dataset.lesson;

  document.querySelectorAll('[data-quiz]').forEach((quiz) => {
    const feedback = quiz.querySelector('[data-feedback]');
    quiz.querySelectorAll('button[data-correct]').forEach((button) => {
      button.addEventListener('click', () => {
        const correct = button.dataset.correct === 'true';
        quiz.querySelectorAll('button[data-correct]').forEach((choice) => {
          choice.classList.remove('correct', 'incorrect');
          choice.setAttribute('aria-pressed', 'false');
        });
        button.classList.add(correct ? 'correct' : 'incorrect');
        button.setAttribute('aria-pressed', 'true');
        feedback.textContent = correct
          ? quiz.dataset.success
          : quiz.dataset.retry;
        feedback.className = `feedback ${correct ? 'ok' : 'no'}`;
      });
    });
  });

  document.querySelectorAll('[data-completion]').forEach((checkbox) => {
    const key = `rope-course:${lessonId}:complete`;
    checkbox.checked = localStorage.getItem(key) === 'yes';
    checkbox.addEventListener('change', () => {
      localStorage.setItem(key, checkbox.checked ? 'yes' : 'no');
    });
  });
})();
