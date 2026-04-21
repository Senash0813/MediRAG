'use client';

import React, { useEffect, useMemo, useRef, useState } from 'react';
import { TypewriterSequence } from '@/components/ui/typewriter';
import { Brain, Heart, Stethoscope, Activity, type LucideIcon } from 'lucide-react';
import { useTheme } from '@/contexts/ThemeContext';
import { landingPipelines } from './content';

const icons: LucideIcon[] = [Brain, Heart, Stethoscope, Activity];

type PipelineEntry = (typeof landingPipelines)[number];

function PipelineCardBody({
  pipeline,
  pipelineIndex,
  isDark,
}: {
  pipeline: PipelineEntry;
  pipelineIndex: number;
  isDark: boolean;
}) {
  const Icon = icons[pipelineIndex];
  return (
    <>
      {/* Top section — identity + description */}
      <div className={`rounded-xl p-6 sm:p-8 ${isDark ? 'bg-white/5' : 'bg-black/[0.03]'}`}>
        <div className="mb-6 flex items-start gap-4 sm:gap-5">
          <div
            className={`flex h-12 w-12 shrink-0 items-center justify-center rounded-xl sm:h-14 sm:w-14 ${
              isDark
                ? 'bg-orange-500/15 text-orange-400 ring-1 ring-orange-500/25'
                : 'bg-orange-500 text-white shadow-sm'
            }`}
          >
            <Icon className="h-6 w-6 sm:h-7 sm:w-7" aria-hidden />
          </div>
          <div>
            <p className="text-xs font-semibold uppercase tracking-wider text-orange-500/80">
              Pipeline
            </p>
            <h3 className="mt-1 text-2xl font-bold tracking-tight sm:text-3xl lg:text-4xl">
              {pipeline.name}
            </h3>
          </div>
        </div>

        <p
          className={`text-base leading-relaxed sm:text-lg ${
            isDark ? 'text-white/85' : 'text-black/80'
          }`}
        >
          {pipeline.description}
        </p>
      </div>

      {/* Bottom section — specialty highlight */}
      <div
        className="mt-3 rounded-xl p-6 sm:p-8"
      >
        <div className="mb-3 flex items-center gap-2">
          <span
            className="inline-flex items-center gap-1.5 rounded-full px-3 py-1.5 text-[11px] font-bold uppercase tracking-wider text-white shadow-[0_2px_12px_rgba(249,115,22,0.45)]"
            style={{ background: 'linear-gradient(135deg, #f97316 0%, #ea580c 100%)' }}
          >
            <span aria-hidden>✦</span> What makes it unique
          </span>
        </div>
        <p
          className={`text-sm leading-relaxed sm:text-base ${
            isDark ? 'text-white/75' : 'text-black/70'
          }`}
        >
          {pipeline.ragFocus}
        </p>
      </div>
    </>
  );
}

export function LandingIntro() {
  const { theme } = useTheme();
  const isDark = theme === 'dark';
  const [activePipeline, setActivePipeline] = useState(0);
  const tabRefs = useRef<(HTMLButtonElement | null)[]>([]);
  const pipelineTabSkipScrollRef = useRef(true);

  useEffect(() => {
    if (pipelineTabSkipScrollRef.current) {
      pipelineTabSkipScrollRef.current = false;
      return;
    }
    tabRefs.current[activePipeline]?.scrollIntoView({
      behavior: 'smooth',
      inline: 'center',
      block: 'nearest',
    });
  }, [activePipeline]);

  const heroSegments = useMemo(
    () => [
      { text: 'Medical intelligence.', className: isDark ? 'text-white' : 'text-black' },
      { text: ' Grounded in evidence.', className: isDark ? 'text-white' : 'text-black' },
      { text: ' Delivered instantly.', className: isDark ? 'text-white' : 'text-black' },
      { text: '+', className: 'text-orange-500 text-[1.35em] leading-none pl-3' },
    ],
    [isDark]
  );

  const bodyText = isDark ? 'text-[#c4c7c5]' : 'text-gray-800';
  const eyebrow = isDark ? 'text-[#8e9196]' : 'text-gray-500';

  return (
    <div className="flex justify-center max-w-[1200px] mx-auto pb-16 px-4 sm:px-6">
      <main id="pipelines" className="max-w-[750px] w-full scroll-mt-28 pt-16">
        <header className="relative mb-10 sm:mb-12 flex min-h-[calc(100dvh-20rem)] flex-col justify-center">

          <div className={`relative text-[13px] font-medium ${eyebrow} mb-5 uppercase tracking-wider`} />
          <h1 className="relative text-5xl sm:text-6xl lg:text-7xl font-bold tracking-tight mb-8 leading-[1.12]">
            <TypewriterSequence
              key={theme}
              segments={heroSegments}
              speed={78}
              delayBetweenSegments={520}
              cursor
              cursorChar="|"
              inline
            />
          </h1>
        </header>

        <article className={`space-y-8 text-lg leading-relaxed ${bodyText}`}>
          <p className="font-bold">
            MediRAG is a purpose-built RAG system for medical question answering, designed to deliver
            accurate, evidence-backed answers you can trust.
          </p>
          <p>
            When it comes to highly specific medical questions, generic LLMs often fall
            short, producing answers you cannot fully rely on. And in medicine, that uncertainty
            is not just frustrating. It is risky.
          </p>
          <p>
            Instead of relying on a single model, MediRAG uses four specialized RAG pipelines, each
            optimized for a distinct medical domain. Every question is intelligently routed to the
            right pipeline, where answers are generated from peer-reviewed evidence.
          </p>
          <p>
            The result? Responses that are precise, context-aware, and clinically grounded.
          </p>
          <p className="font-bold italic">
            No hallucinations. No uncertainty. Just medical answers, done right.
          </p>
        </article>

        <section className="mt-14 sm:mt-16" aria-labelledby="pipelines-tabs-heading">
          <h2 id="pipelines-tabs-heading" className="sr-only">
            Pipeline introductions
          </h2>
          <div className="-mx-4 px-4 pb-4 sm:mx-0 sm:px-0">
            <div
              className={`w-full rounded-full p-1.5 sm:p-2 ${
                isDark
                  ? 'border border-[#3c4043] bg-[#1e1f20]/80 backdrop-blur-sm'
                  : 'bg-gray-100/80 backdrop-blur-sm'
              }`}
            >
              <div
                className="flex w-full gap-1 overflow-x-auto scroll-smooth scrollbar-hide sm:overflow-visible"
                role="tablist"
                aria-label="Choose a clinical pipeline"
              >
                {landingPipelines.map((pipe, i) => {
                  const selected = activePipeline === i;
                  return (
                    <button
                      key={pipe.name}
                      ref={(el) => {
                        tabRefs.current[i] = el;
                      }}
                      type="button"
                      role="tab"
                      aria-selected={selected}
                      id={`pipeline-tab-${i}`}
                      aria-controls="pipeline-panel"
                      onClick={() => setActivePipeline(i)}
                      className={`rounded-full px-3 py-2 text-[13px] font-medium transition-[background-color,color,transform] duration-300 ease-out max-sm:shrink-0 max-sm:whitespace-nowrap sm:min-w-0 sm:flex-1 sm:basis-0 sm:px-3 sm:py-2.5 sm:text-center sm:text-[14px] sm:leading-snug lg:text-[15px] ${
                        isDark
                          ? selected
                            ? 'scale-[1.02] bg-white text-black shadow-sm'
                            : 'scale-100 bg-transparent text-[#c4c7c5] hover:text-white'
                          : selected
                            ? 'scale-[1.02] bg-black text-white shadow-sm'
                            : 'scale-100 bg-transparent text-gray-700 hover:text-black'
                      }`}
                    >
                      {pipe.name}
                    </button>
                  );
                })}
              </div>
            </div>
          </div>

          <div
            id="pipeline-panel"
            role="tabpanel"
            aria-labelledby={`pipeline-tab-${activePipeline}`}
            className={`rounded-2xl border-0 p-4 sm:p-5 transition-[background-color,box-shadow] duration-300 ease-out ${
              isDark
                ? 'bg-[#1c1c1e]/85 backdrop-blur-sm text-white shadow-[0_12px_40px_-8px_rgba(0,0,0,0.65)]'
                : 'bg-[#ececec]/80 backdrop-blur-sm text-black shadow-[0_10px_36px_-10px_rgba(0,0,0,0.22)]'
            }`}
          >
            {/*
              All bodies occupy the same grid cell so row height = tallest pipeline (fixed visual height per tab).
              Inactive layers stay visibility:hidden but still participate in layout sizing.
            */}
            <div className="grid grid-cols-1">
              {landingPipelines.map((p, i) => {
                const isActive = i === activePipeline;
                return (
                  <div
                    key={p.name}
                    className={`col-start-1 row-start-1 w-full min-w-0 ${
                      isActive
                        ? 'visible relative z-10'
                        : 'invisible pointer-events-none'
                    }`}
                    aria-hidden={!isActive}
                  >
                    <div
                      key={isActive ? String(activePipeline) : `slot-${i}`}
                      className={isActive ? 'animate-in fade-in' : undefined}
                      style={
                        isActive
                          ? {
                              animationDuration: '0.4s',
                              animationTimingFunction: 'cubic-bezier(0.22, 1, 0.36, 1)',
                            }
                          : undefined
                      }
                    >
                      <PipelineCardBody
                        pipeline={p}
                        pipelineIndex={i}
                        isDark={isDark}
                      />
                    </div>
                  </div>
                );
              })}
            </div>
          </div>
        </section>

        <div className={`mt-16 h-px w-full ${isDark ? 'bg-[#3c4043]' : 'bg-gray-200'}`} />
      </main>
    </div>
  );
}
