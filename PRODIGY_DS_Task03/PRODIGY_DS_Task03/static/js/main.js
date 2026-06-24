// main.js — Bank Marketing DT UI helpers

async function retrain() {
  const toast = document.getElementById('retrain-toast');
  if (toast) {
    toast.classList.remove('hidden');
    toast.textContent = '⟳ Retraining model...';
  }
  try {
    const res = await fetch('/api/retrain', { method: 'POST' });
    const data = await res.json();
    if (toast) {
      toast.textContent = `✅ Done! Accuracy: ${data.stats.accuracy}%`;
      setTimeout(() => {
        toast.classList.add('hidden');
        location.reload();
      }, 2000);
    }
  } catch (e) {
    if (toast) {
      toast.textContent = '❌ Retrain failed: ' + e.message;
      setTimeout(() => toast.classList.add('hidden'), 3000);
    }
  }
}

// Animate stat values on load
document.addEventListener('DOMContentLoaded', () => {
  document.querySelectorAll('.stat-value').forEach(el => {
    const target = parseFloat(el.textContent);
    if (isNaN(target)) return;
    const suffix = el.textContent.includes('%') ? '%' : '';
    const isFloat = el.textContent.includes('.');
    let start = 0;
    const step = target / 40;
    const timer = setInterval(() => {
      start += step;
      if (start >= target) { start = target; clearInterval(timer); }
      el.textContent = (isFloat ? start.toFixed(1) : Math.floor(start)) + suffix;
    }, 25);
  });
});
