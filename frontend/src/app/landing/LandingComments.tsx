'use client';

import React, { useMemo, useState } from 'react';
import { ChevronLeft, ChevronRight } from 'lucide-react';
import { useTheme } from '@/contexts/ThemeContext';
import { landingTestimonials } from './content';
import { landingCardSurfaceClass, landingMutedSubtitleClass } from './theme';

const COMMENTS_PER_PAGE = 4;

export function LandingComments() {
  const { theme } = useTheme();
  const isDark = theme === 'dark';
  const mode = isDark ? 'dark' : 'light';
  const card = landingCardSurfaceClass(mode);
  const subtitle = landingMutedSubtitleClass(mode);

  const needsCarousel = landingTestimonials.length > COMMENTS_PER_PAGE;
  const pageCount = Math.ceil(landingTestimonials.length / COMMENTS_PER_PAGE);
  const [page, setPage] = useState(0);

  const visibleTestimonials = useMemo(() => {
    if (!needsCarousel) {
      return landingTestimonials;
    }
    const start = page * COMMENTS_PER_PAGE;
    return landingTestimonials.slice(start, start + COMMENTS_PER_PAGE);
  }, [needsCarousel, page]);

  const navButtonClass = `flex h-10 w-10 shrink-0 items-center justify-center rounded-full border transition-[opacity,background-color,color] disabled:pointer-events-none disabled:opacity-35 ${
    isDark
      ? 'border-[#3c4043] bg-[#1e1f20] text-[#e3e3e3] hover:bg-[#282a2c]'
      : 'border-gray-200 bg-white text-gray-900 shadow-sm hover:bg-gray-50'
  }`;

  return (
    <section
      id="voices"
      className="max-w-[1000px] mx-auto px-4 sm:px-6 pb-20 scroll-mt-28"
      aria-labelledby="voices-heading"
    >
      <div className="mx-auto w-full max-w-[750px] mb-10">
        <h2
          id="voices-heading"
          className={`text-center text-3xl sm:text-4xl font-semibold tracking-tight mb-4 ${
            isDark ? 'text-[#e3e3e3]' : 'text-[#050505]'
          }`}
        >
          Voices of early users
        </h2>
        <p className={`text-lg font-medium leading-relaxed ${subtitle}`}>
          Comments from industry partners and clinical early users—shared to highlight what stood out
          in pilots, not as endorsements of any specific deployment.
        </p>
      </div>

      <div
        className={`mx-auto flex w-full items-center ${
          needsCarousel ? 'max-w-[1000px] gap-2 sm:gap-3' : 'max-w-[952px] justify-center'
        }`}
      >
        {needsCarousel ? (
          <button
            type="button"
            className={navButtonClass}
            aria-label="Previous comments"
            disabled={page <= 0}
            onClick={() => setPage((p) => Math.max(0, p - 1))}
          >
            <ChevronLeft className="h-5 w-5" aria-hidden />
          </button>
        ) : null}

        <div
          className={`grid min-w-0 grid-cols-1 justify-items-center gap-5 sm:grid-cols-2 sm:justify-items-stretch ${
            needsCarousel ? 'flex-1 max-w-[952px]' : 'w-full max-w-[952px]'
          }`}
          aria-live="polite"
          aria-atomic="true"
        >
          {visibleTestimonials.map((t) => (
            <blockquote
              key={t.name}
              className={`flex h-[300px] w-full max-w-[466px] flex-col overflow-hidden rounded-2xl p-6 sm:h-[320px] sm:max-w-none sm:p-8 ${card}`}
            >
              <p
                className={`min-h-0 flex-1 overflow-y-auto text-[16px] sm:text-lg leading-relaxed font-medium ${
                  isDark ? 'text-[#e3e3e3]' : 'text-gray-800'
                } mb-4`}
              >
                “{t.quote}”
              </p>
              <footer className="shrink-0 pt-2">
                <div
                  className={`font-semibold text-[15px] ${
                    isDark ? 'text-white' : 'text-gray-900'
                  }`}
                >
                  {t.name}
                </div>
                <div className={`text-[14px] font-medium mt-1 ${subtitle}`}>
                  {t.role}
                  {t.org ? ` · ${t.org}` : ''}
                </div>
              </footer>
            </blockquote>
          ))}
        </div>

        {needsCarousel ? (
          <button
            type="button"
            className={navButtonClass}
            aria-label="Next comments"
            disabled={page >= pageCount - 1}
            onClick={() => setPage((p) => Math.min(pageCount - 1, p + 1))}
          >
            <ChevronRight className="h-5 w-5" aria-hidden />
          </button>
        ) : null}
      </div>
    </section>
  );
}
