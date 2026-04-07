'use client';

import React from 'react';
import Link from 'next/link';
import { landingGradients } from './theme';

const linkClass = 'text-gray-400 hover:text-white transition-colors';

export function LandingFooter() {
  const year = new Date().getFullYear();

  return (
    <footer
      id="developers"
      className="bg-black text-white pt-14 sm:pt-16 pb-8 px-4 sm:px-6 mt-4 scroll-mt-28"
    >
      <div className="max-w-[1200px] mx-auto grid grid-cols-2 md:grid-cols-4 gap-x-8 gap-y-10 text-[15px]">
        <div className="col-span-2 md:col-span-1">
          <div className="font-semibold text-lg tracking-tight mb-3 text-white">
            <span className={landingGradients.headlineTextClass}>Medi</span>
            <span className="text-white">RAG</span>
          </div>
          <p className="text-gray-500 text-sm font-medium leading-relaxed max-w-xs">
            Four clinical RAG pipelines. One place to explore evidence-grounded answers.
          </p>
        </div>
        <div>
          <div className="font-semibold mb-4 text-gray-100">Product</div>
          <ul className="space-y-3 text-gray-400 font-medium">
            <li>
              <a href="#pipelines" className={linkClass}>
                Pipelines
              </a>
            </li>
            <li>
              <a href="#voices" className={linkClass}>
                Voices
              </a>
            </li>
            <li>
              <Link href="/" className={linkClass}>
                Open app
              </Link>
            </li>
          </ul>
        </div>
        <div>
          <div className="font-semibold mb-4 text-gray-100">Domains</div>
          <ul className="space-y-3 text-gray-400 font-medium">
            <li>
              <span className="text-gray-500">Neurosciences</span>
            </li>
            <li>
              <span className="text-gray-500">Cardiovascular</span>
            </li>
            <li>
              <span className="text-gray-500">Internal medicine</span>
            </li>
            <li>
              <span className="text-gray-500">Primary care & mental health</span>
            </li>
          </ul>
        </div>
        <div>
          <div className="font-semibold mb-4 text-gray-100">Notice</div>
          <p className="text-gray-500 text-sm font-medium leading-relaxed">
            Not a medical device. Outputs are for research and education; always verify against
            institutional policy and primary literature.
          </p>
        </div>
      </div>
      <div className="max-w-[1200px] mx-auto mt-14 pt-8 border-t border-gray-800 text-sm text-gray-500 flex flex-col md:flex-row justify-between items-center gap-4 font-medium">
        <div>MediRAG © {year}</div>
        <div className="flex flex-wrap justify-center gap-6">
          <Link href="/" className="hover:text-white transition-colors">
            Application
          </Link>
          <Link href="/landing" className="hover:text-white transition-colors">
            Landing
          </Link>
        </div>
      </div>
    </footer>
  );
}
