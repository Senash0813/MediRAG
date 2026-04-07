'use client';

import React from 'react';
import { useTheme } from '@/contexts/ThemeContext';
import { LandingHeader } from './LandingHeader';
import { LandingIntro } from './LandingIntro';
import { LandingComments } from './LandingComments';
import { LandingFooter } from './LandingFooter';

export function LandingView() {
  const { theme } = useTheme();
  const isDark = theme === 'dark';

  return (
    <div
      className={`min-h-screen w-full font-sans scroll-smooth ${
        isDark
          ? 'bg-[#131314] text-[#e3e3e3] selection:bg-white/20 selection:text-white'
          : 'bg-[#fcfcfc] text-[#050505] selection:bg-black selection:text-white'
      }`}
    >
      <LandingHeader />
      <LandingIntro />
      <LandingComments />
      <LandingFooter />
    </div>
  );
}
