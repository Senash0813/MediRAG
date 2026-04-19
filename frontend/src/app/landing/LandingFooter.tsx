'use client';

import React from 'react';

export function LandingFooter() {
  const year = new Date().getFullYear();

  return (
    <footer
      id="footer"
      className="mt-4 scroll-mt-28 border-t border-gray-800 bg-black px-4 py-2.5 sm:px-6 sm:py-0"
    >
      <div className="mx-auto flex w-full max-w-[1200px] flex-col items-center justify-center gap-2 py-2 text-center md:flex-row md:flex-nowrap md:gap-8 md:py-3">
        <div className="shrink-0 text-lg font-bold tracking-tight text-white sm:text-xl">MediRAG</div>
        <p className="shrink-0 text-[11px] font-medium text-gray-500 sm:text-xs">© {year} MediRAG</p>
        <p className="min-w-0 max-w-2xl text-[11px] leading-snug text-gray-500 sm:max-w-none lg:text-xs">
          MediRAG can still make mistakes—always verify answers against primary sources and your own
          judgment before clinical or educational use.
        </p>
      </div>
    </footer>
  );
}
