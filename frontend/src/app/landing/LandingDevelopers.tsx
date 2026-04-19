'use client';

import React from 'react';
import Image from 'next/image';
import { useTheme } from '@/contexts/ThemeContext';
import { landingDevelopers } from './content';

export function LandingDevelopers() {
  const { theme } = useTheme();
  const isDark = theme === 'dark';
  const bodyText = isDark ? 'text-[#a8abaa]' : 'text-gray-600';

  return (
    <section
      id="developers"
      className="scroll-mt-28 px-4 pb-16 pt-16 sm:px-6 sm:pb-20 sm:pt-20"
      aria-labelledby="developers-heading"
    >
      <div
        className={`mx-auto h-px max-w-[750px] ${isDark ? 'bg-[#3c4043]' : 'bg-gray-200'}`}
        aria-hidden
      />
      <div className="mx-auto mt-16 flex max-w-[1000px] flex-col gap-10 sm:mt-20 lg:flex-row lg:items-center lg:gap-14">
        <div
          className={`relative w-full shrink-0 overflow-hidden rounded-2xl border lg:max-w-[min(100%,480px)] lg:flex-1 ${
            isDark
              ? 'border-[#3c4043] bg-[#1e1f20] shadow-[0_12px_40px_-8px_rgba(0,0,0,0.45)]'
              : 'border-gray-200 bg-gray-50 shadow-[0_10px_36px_-10px_rgba(0,0,0,0.12)]'
          }`}
        >
          <Image
            src="/landing/developers-team.svg"
            alt="MediRAG development team"
            width={800}
            height={520}
            className="h-auto w-full object-cover"
            sizes="(max-width: 1024px) 100vw, 480px"
            priority={false}
          />
        </div>

        <div className="min-w-0 flex-1 lg:max-w-[480px]">
          <h2
            id="developers-heading"
            className={`mb-5 text-3xl font-semibold tracking-tight sm:text-4xl ${
              isDark ? 'text-[#e3e3e3]' : 'text-[#050505]'
            }`}
          >
            {landingDevelopers.title}
          </h2>
          <div className={`space-y-4 text-lg font-medium leading-relaxed ${bodyText}`}>
            {landingDevelopers.paragraphs.map((p, i) => (
              <p key={i}>{p}</p>
            ))}
          </div>
        </div>
      </div>
    </section>
  );
}
