/* Startup splash: plays splash-screen.gif once, then fades out. Runs before the app loads. */
(() => {
  const PLAY_MS = 3100;  /* about one loop of the GIF (3.12 s) */
  const MAX_MS = 6000;   /* give up waiting if the GIF is slow to load */
  const splash = document.getElementById('splash');
  const img = document.getElementById('splashImg');
  let done = false;

  const hide = () => {
    if (done) return;
    done = true;
    splash.classList.add('out');
    setTimeout(() => splash.remove(), 500);
  };
  const arm = () => setTimeout(hide, PLAY_MS);

  if (img.complete && img.naturalWidth) arm();
  else { img.onload = arm; img.onerror = hide; }
  setTimeout(hide, MAX_MS);
  splash.onclick = hide;  /* tap to skip */
})();
