/**
 * Scrolls the viewport to an element with id matching the hash.
 * Optional leading delay, then linear motion at a roughly constant velocity (px/s), so short
 * hops (e.g. Voices → Developers) are not artificially stretched to the same duration as long ones.
 * Respects the target element's CSS scroll-margin (e.g. Tailwind scroll-mt-*).
 */

const DEFAULT_DELAY_MS = 220;
/** Target scroll speed; duration = distance / speed (clamped). */
const PIXELS_PER_SECOND = 820;
const MIN_DURATION_MS = 380;
const MAX_DURATION_MS = 1600;

function parseScrollMarginTop(el: HTMLElement): number {
  const raw = getComputedStyle(el).scrollMarginTop;
  if (!raw || raw === 'auto') return 0;
  const n = parseFloat(raw);
  if (raw.endsWith('rem')) {
    const rootPx = parseFloat(getComputedStyle(document.documentElement).fontSize) || 16;
    return n * rootPx;
  }
  return Number.isFinite(n) ? n : 0;
}

export function smoothScrollToHash(
  hash: string,
  options?: { delayMs?: number; /** Override auto duration from distance */ durationMs?: number }
): void {
  const id = hash.startsWith('#') ? hash.slice(1) : hash;
  if (!id) return;

  const el = document.getElementById(id);
  if (!el) return;

  if (window.matchMedia('(prefers-reduced-motion: reduce)').matches) {
    const marginTop = parseScrollMarginTop(el);
    const rect = el.getBoundingClientRect();
    const targetY = window.scrollY + rect.top - marginTop;
    window.scrollTo(0, targetY);
    try {
      history.pushState(null, '', `#${id}`);
    } catch {
      /* ignore */
    }
    return;
  }

  const delayMs = options?.delayMs ?? DEFAULT_DELAY_MS;

  window.setTimeout(() => {
    const marginTop = parseScrollMarginTop(el);
    const rect = el.getBoundingClientRect();
    const startY = window.scrollY;
    const targetY = startY + rect.top - marginTop;
    const distance = targetY - startY;

    const autoDuration = (Math.abs(distance) / PIXELS_PER_SECOND) * 1000;
    const durationMs =
      options?.durationMs ??
      Math.min(MAX_DURATION_MS, Math.max(MIN_DURATION_MS, autoDuration));

    const finish = () => {
      window.scrollTo(0, targetY);
      try {
        history.pushState(null, '', `#${id}`);
      } catch {
        /* ignore */
      }
    };

    if (Math.abs(distance) < 0.5) {
      finish();
      return;
    }

    let startTime: number | null = null;

    const step = (now: number) => {
      if (startTime === null) startTime = now;
      const elapsed = now - startTime;
      const t = Math.min(1, elapsed / durationMs);
      const y = startY + distance * t;
      window.scrollTo(0, y);
      if (t < 1) {
        requestAnimationFrame(step);
      } else {
        finish();
      }
    };

    requestAnimationFrame(step);
  }, delayMs);
}
