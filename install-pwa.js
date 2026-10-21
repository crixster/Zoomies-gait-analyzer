let deferredPrompt = null;

const installContainer = document.getElementById('pwaInstallContainer');
const installBtn = document.getElementById('pwaInstallBtn');

// Hide if already installed
if (
  window.matchMedia('(display-mode: standalone)').matches ||
  window.navigator.standalone
) {
  installContainer?.classList.add('hidden');
}

window.addEventListener('beforeinstallprompt', (e) => {
  e.preventDefault();

  deferredPrompt = e;

  installContainer?.classList.remove('hidden');
});

installBtn?.addEventListener('click', async () => {
  if (!deferredPrompt) return;

  deferredPrompt.prompt();

  await deferredPrompt.userChoice;

  deferredPrompt = null;

  installContainer?.classList.add('hidden');
});

window.addEventListener('appinstalled', () => {
  installContainer?.classList.add('hidden');
  deferredPrompt = null;

  console.log('PWA installed');
});
