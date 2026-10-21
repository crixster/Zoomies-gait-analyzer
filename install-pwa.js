let deferredPrompt = null;

const installContainer = document.getElementById('pwaInstallContainer');
const installBtn = document.getElementById('pwaInstallBtn');

// Already installed?
if (
    window.matchMedia('(display-mode: standalone)').matches ||
    window.navigator.standalone
) {
    installContainer?.classList.add('hidden');
}

window.addEventListener('beforeinstallprompt', (e) => {
    e.preventDefault();

    deferredPrompt = e;

    // Only show button if not installed
    if (
        !window.matchMedia('(display-mode: standalone)').matches &&
        !window.navigator.standalone
    ) {
        installContainer?.classList.remove('hidden');
    }
});

installBtn?.addEventListener('click', async () => {
    if (!deferredPrompt) return;

    deferredPrompt.prompt();

    const result = await deferredPrompt.userChoice;

    if (result.outcome === 'accepted') {
        installContainer?.classList.add('hidden');
    }

    deferredPrompt = null;
});

window.addEventListener('appinstalled', () => {
    console.log('PWA installed');

    installContainer?.classList.add('hidden');
    deferredPrompt = null;
});
