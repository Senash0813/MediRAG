/**
 * Landing page design tokens — aligned with the main app welcome UI
 * (ChatArea headline, page shell, globals.css, ClusterCard accents).
 * Uses Geist from root layout via CSS variables on <body>.
 */

export type LandingThemeMode = 'light' | 'dark';

/** Raw colors for inline styles or documentation */
export const landingColors = {
  light: {
    pageBg: '#f4f4f5',
    foreground: '#18181b',
    muted: '#9ca3af',
    mutedStrong: '#6b7280',
    border: '#e5e7eb',
    card: '#ffffff',
    cardMuted: '#f9fafb',
  },
  dark: {
    pageBg: '#131314',
    foreground: '#e3e3e3',
    muted: '#444746',
    mutedIcon: '#8e9196',
    border: '#3c4043',
    card: '#1e1f20',
    cardElevated: '#282a2c',
  },
} as const;

/** Google-style gradient used on “Welcome to MediRAG” */
export const landingGradients = {
  headlineFrom: '#4285f4',
  headlineVia: '#9b72cb',
  headlineTo: '#d96570',
  /** Tailwind utility chain for gradient text (same as ChatArea) */
  headlineTextClass:
    'bg-gradient-to-r from-[#4285f4] via-[#9b72cb] to-[#d96570] bg-clip-text text-transparent',
} as const;

export const landingFonts = {
  /** Matches layout.tsx Geist + globals @theme */
  sans: 'var(--font-geist-sans), ui-sans-serif, system-ui, sans-serif',
  mono: 'var(--font-geist-mono), ui-monospace, monospace',
} as const;

/** Typography scale — weights and tracking match ChatArea welcome */
export const landingTypography = {
  display: 'text-5xl sm:text-6xl font-medium tracking-tight',
  subtitle: 'text-2xl sm:text-3xl font-medium tracking-tight',
  sectionTitle: 'text-2xl sm:text-3xl font-medium tracking-tight',
  lead: 'text-base sm:text-lg font-medium leading-relaxed',
  body: 'text-[15px] font-medium leading-relaxed tracking-tight',
  caption: 'text-sm font-medium',
  cardTitle: 'text-base font-medium tracking-tight',
  cardBody: 'text-[12.5px] font-medium leading-relaxed',
} as const;

/** Page shell (mirrors home: font-sans + light/dark backgrounds) */
export const landingShell = {
  root: 'min-h-screen font-sans antialiased',
  light: 'bg-background text-gray-900',
  dark: 'bg-[#131314] text-[#e3e3e3]',
} as const;

/** Content width — same as ChatArea max width */
export const landingLayout = {
  section: 'w-full max-w-4xl mx-auto px-6',
  wideSection: 'w-full max-w-6xl mx-auto px-6',
} as const;

/**
 * Pipeline / cluster accents — same order and feel as ClusterCard (1–4).
 * Use with template literals or split for ring-offset (dark vs light).
 */
export const landingPipelineAccents = [
  {
    border: 'border-blue-500/30 hover:border-blue-500/60',
    bg: 'bg-blue-500/5',
    icon: 'text-blue-400',
    ringOffsetLight: 'ring-offset-white',
    ringOffsetDark: 'ring-offset-[#131314]',
  },
  {
    border: 'border-purple-500/30 hover:border-purple-500/60',
    bg: 'bg-purple-500/5',
    icon: 'text-purple-400',
    ringOffsetLight: 'ring-offset-white',
    ringOffsetDark: 'ring-offset-[#131314]',
  },
  {
    border: 'border-green-500/30 hover:border-green-500/60',
    bg: 'bg-green-500/5',
    icon: 'text-green-400',
    ringOffsetLight: 'ring-offset-white',
    ringOffsetDark: 'ring-offset-[#131314]',
  },
  {
    border: 'border-orange-500/30 hover:border-orange-500/60',
    bg: 'bg-orange-500/5',
    icon: 'text-orange-400',
    ringOffsetLight: 'ring-offset-white',
    ringOffsetDark: 'ring-offset-[#131314]',
  },
] as const;

/** Muted subtitle color class by theme (ChatArea h4) */
export function landingMutedSubtitleClass(mode: LandingThemeMode): string {
  return mode === 'dark' ? 'text-[#444746]' : 'text-gray-400';
}

/** Testimonial / elevated surface — chat bubble–adjacent */
export function landingCardSurfaceClass(mode: LandingThemeMode): string {
  return mode === 'dark'
    ? 'bg-[#282a2c] border border-[#3c4043]'
    : 'bg-gray-50 border border-gray-200';
}

/** Icon chip background behind Lucide icons (ClusterCard) */
export function landingIconChipClass(mode: LandingThemeMode): string {
  return mode === 'dark' ? 'bg-[#1e1f20]' : 'bg-gray-100';
}
