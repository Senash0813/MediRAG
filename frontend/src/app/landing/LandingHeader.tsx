'use client';

import React, { useState } from 'react';
import Link from 'next/link';
import { Menu, X, ArrowUpRight } from 'lucide-react';
import { useTheme } from '@/contexts/ThemeContext';

type NavKey = 'pipelines' | 'voices' | 'developers';

const navLinkClass =
  'text-black text-[15px] font-medium pb-0.5 border-b-2 border-transparent transition-colors hover:border-black';

function navLinkActiveClass(active: NavKey | null, key: NavKey): string {
  return active === key ? 'border-black' : '';
}

export function LandingHeader() {
  const [open, setOpen] = useState(false);
  const [activeNav, setActiveNav] = useState<NavKey | null>(null);
  const { theme } = useTheme();
  const isDark = theme === 'dark';

  const onNavClick = (key: NavKey) => {
    setActiveNav(key);
    setOpen(false);
  };

  const mobileNavLink = (key: NavKey) =>
    isDark
      ? `block w-full text-left pb-3 border-b-2 transition-colors ${
          activeNav === key
            ? 'border-white text-white'
            : 'border-transparent text-[#e3e3e3] hover:border-white'
        }`
      : `block w-full text-left pb-3 border-b-2 transition-colors ${
          activeNav === key
            ? 'border-black text-black'
            : 'border-transparent text-black hover:border-black'
        }`;

  const navBarClass = isDark
    ? 'bg-[#131314]/80 backdrop-blur-md'
    : 'bg-[#fcfcfc]/80 backdrop-blur-md';

  return (
    <>
      <nav className={`fixed top-0 left-0 right-0 z-50 ${navBarClass}`}>
        <div className="max-w-[1400px] mx-auto px-4 sm:px-6 h-16 flex items-center justify-between">
          <div className="flex items-center gap-10 lg:gap-12">
            <Link
              href="/landing"
              className="text-black font-bold text-lg sm:text-xl tracking-tight"
            >
              MediRAG
            </Link>
            <div className="hidden md:flex items-center gap-8">
              <a
                href="#pipelines"
                className={`${navLinkClass} ${navLinkActiveClass(activeNav, 'pipelines')}`}
                onClick={() => onNavClick('pipelines')}
              >
                What is it?
              </a>
              <a
                href="#voices"
                className={`${navLinkClass} ${navLinkActiveClass(activeNav, 'voices')}`}
                onClick={() => onNavClick('voices')}
              >
                Voices
              </a>
              <a
                href="#developers"
                className={`${navLinkClass} ${navLinkActiveClass(activeNav, 'developers')}`}
                onClick={() => onNavClick('developers')}
              >
                Meet Developers
              </a>
            </div>
          </div>

          <div className="hidden md:flex items-center text-[15px] font-medium">
            <Link
              href="/"
              className="rounded-lg px-4 py-2 flex items-center gap-1 transition-colors bg-black text-white hover:bg-gray-900"
            >
              Open app
              <ArrowUpRight className="w-4 h-4 shrink-0" aria-hidden />
            </Link>
          </div>

          <button
            type="button"
            className="md:hidden p-2 rounded-lg text-black"
            aria-expanded={open}
            aria-label={open ? 'Close menu' : 'Open menu'}
            onClick={() => setOpen((v) => !v)}
          >
            {open ? <X className="w-6 h-6" /> : <Menu className="w-6 h-6" />}
          </button>
        </div>
      </nav>

      {open && (
        <div
          className={`fixed inset-0 top-16 z-40 md:hidden overflow-y-auto ${
            isDark ? 'bg-[#131314]' : 'bg-white'
          }`}
        >
          <div className="flex flex-col p-6 gap-4 text-lg font-medium">
            <a
              href="#pipelines"
              className={mobileNavLink('pipelines')}
              onClick={() => onNavClick('pipelines')}
            >
              Pipelines
            </a>
            <a
              href="#voices"
              className={mobileNavLink('voices')}
              onClick={() => onNavClick('voices')}
            >
              Voices
            </a>
            <a
              href="#developers"
              className={mobileNavLink('developers')}
              onClick={() => onNavClick('developers')}
            >
              Meet Developers
            </a>
            <div className="pt-2">
              <Link
                href="/"
                onClick={() => setOpen(false)}
                className="rounded-lg px-4 py-3 flex justify-center items-center gap-1 bg-black text-white hover:bg-gray-900"
              >
                Open app
                <ArrowUpRight className="w-5 h-5 shrink-0" />
              </Link>
            </div>
          </div>
        </div>
      )}
    </>
  );
}
