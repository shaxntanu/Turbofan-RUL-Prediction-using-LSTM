const slides = [...document.querySelectorAll('.slide')];
const previous = document.querySelector('#previous');
const next = document.querySelector('#next');
const count = document.querySelector('#slide-count');
const fullscreen = document.querySelector('#fullscreen');
let current = 0;

function showSlide(index, updateHash = true) {
  current = Math.max(0, Math.min(index, slides.length - 1));
  slides.forEach((slide, i) => { slide.hidden = i !== current; });
  count.textContent = `${String(current + 1).padStart(2, '0')} / ${String(slides.length).padStart(2, '0')}`;
  previous.disabled = current === 0;
  next.disabled = current === slides.length - 1;
  if (updateHash) history.replaceState(null, '', `#${slides[current].id}`);
  window.scrollTo(0, 0);
}

function indexFromHash() {
  const index = slides.findIndex((slide) => `#${slide.id}` === window.location.hash);
  return index < 0 ? 0 : index;
}

previous.addEventListener('click', () => showSlide(current - 1));
next.addEventListener('click', () => showSlide(current + 1));
document.querySelector('#print').addEventListener('click', () => window.print());
fullscreen.addEventListener('click', async () => {
  try {
    if (document.fullscreenElement) await document.exitFullscreen();
    else await document.documentElement.requestFullscreen();
  } catch {
    fullscreen.textContent = 'Unavailable';
  }
});
document.addEventListener('fullscreenchange', () => {
  fullscreen.textContent = document.fullscreenElement ? 'Exit fullscreen' : 'Fullscreen';
});
document.addEventListener('keydown', (event) => {
  if (event.altKey || event.ctrlKey || event.metaKey || /^(INPUT|TEXTAREA|SELECT)$/.test(event.target.tagName)) return;
  if (event.key === ' ' && /^(BUTTON|A)$/.test(event.target.tagName)) return;
  if (['ArrowRight', 'PageDown', ' '].includes(event.key)) { event.preventDefault(); showSlide(current + 1); }
  else if (['ArrowLeft', 'PageUp'].includes(event.key)) { event.preventDefault(); showSlide(current - 1); }
  else if (event.key === 'Home') { event.preventDefault(); showSlide(0); }
  else if (event.key === 'End') { event.preventDefault(); showSlide(slides.length - 1); }
});
window.addEventListener('hashchange', () => showSlide(indexFromHash(), false));
showSlide(indexFromHash(), false);
