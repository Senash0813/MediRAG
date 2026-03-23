'use client';

import React from 'react';
import { useTheme } from '@/contexts/ThemeContext';

interface FormattedAnswerProps {
  answer: string;
  isCluster4?: boolean;
}

export const FormattedAnswer = ({ answer, isCluster4 = false, verificationLevel }: FormattedAnswerProps & { verificationLevel?: number }) => {
  const { theme } = useTheme();
	const vLevel = verificationLevel ?? 1;

  // If not cluster 4, just render plain text with pre-wrap
  if (!isCluster4) {
    return (
      <p className={`text-[15.5px] font-medium tracking-tight leading-relaxed whitespace-pre-wrap ${
        theme === 'dark' ? 'text-[#e3e3e3]' : 'text-gray-900'
      }`}>
        {answer}
      </p>
    );
  }

  // Parse cluster 4 response format - split by section headers
  const parts = answer.split(/\*\*(Evidence Summary|Limitations):\*\*/);
  
  return (
    <div className="space-y-3">
      {/* Medical Evidence Warning (Based on Verification Level) */}
      {vLevel === 1 ? (
        <div className="bg-red-50 border-l-4 border-red-500 rounded-r-lg p-4 space-y-2">
          <div className="flex items-start gap-3">
            <svg className="w-5 h-5 text-red-600 flex-shrink-0 mt-0.5" fill="currentColor" viewBox="0 0 20 20">
              <path
                fillRule="evenodd"
                d="M8.257 3.099c.765-1.36 2.722-1.36 3.486 0l5.58 9.92c.75 1.334-.213 2.98-1.742 2.98H4.42c-1.53 0-2.493-1.646-1.743-2.98l5.58-9.92zM11 13a1 1 0 11-2 0 1 1 0 012 0zm-1-8a1 1 0 00-1 1v3a1 1 0 002 0V6a1 1 0 00-1-1z"
                clipRule="evenodd"
              />
            </svg>
            <div className="flex-1">
              <h4 className="text-sm font-semibold text-red-900 mb-1">⚠️ Limited Evidence - Clinical Consultation Required</h4>
              <p className="text-xs text-red-800 leading-relaxed">
                <strong>Low verification level (L1)</strong> indicates insufficient or low-quality evidence. This response is based on limited medical literature and should NOT be used for clinical decision-making. Always consult with a qualified healthcare professional before making any medical decisions.
              </p>
            </div>
          </div>
        </div>
      ) : vLevel === 2 ? (
        <div className="bg-amber-50 border-l-4 border-amber-500 rounded-r-lg p-4 space-y-2">
          <div className="flex items-start gap-3">
            <svg className="w-5 h-5 text-amber-600 flex-shrink-0 mt-0.5" fill="currentColor" viewBox="0 0 20 20">
              <path
                fillRule="evenodd"
                d="M18 10a8 8 0 11-16 0 8 8 0 0116 0zm-7-4a1 1 0 11-2 0 1 1 0 012 0zM9 9a1 1 0 000 2v3a1 1 0 001 1h1a1 1 0 100-2v-3a1 1 0 00-1-1H9z"
                clipRule="evenodd"
              />
            </svg>
            <div className="flex-1">
              <h4 className="text-sm font-semibold text-amber-900 mb-1"> Moderate Evidence - Professional Judgment Required</h4>
              <p className="text-xs text-amber-800 leading-relaxed">
                <strong>Moderate verification level (L2)</strong> indicates emerging or mixed-quality evidence. Clinical judgment and professional consultation are essential. Individual patient factors may significantly influence treatment decisions.
              </p>
            </div>
          </div>
        </div>
      ) : vLevel === 3 ? (
        <div className="bg-blue-50 border-l-4 border-blue-500 rounded-r-lg p-4 space-y-2">
          <div className="flex items-start gap-3">
            <svg className="w-5 h-5 text-blue-600 flex-shrink-0 mt-0.5" fill="currentColor" viewBox="0 0 20 20">
              <path
                fillRule="evenodd"
                d="M18 10a8 8 0 11-16 0 8 8 0 0116 0zm-7-4a1 1 0 11-2 0 1 1 0 012 0zM9 9a1 1 0 000 2v3a1 1 0 001 1h1a1 1 0 100-2v-3a1 1 0 00-1-1H9z"
                clipRule="evenodd"
              />
            </svg>
            <div className="flex-1">
              <h4 className="text-sm font-semibold text-blue-900 mb-1"> Good Evidence Base - Clinical Application Recommended</h4>
              <p className="text-xs text-blue-800 leading-relaxed">
                <strong>High verification level (L3)</strong> indicates good-quality supporting evidence from peer-reviewed sources. While evidence is robust, individual patient circumstances, contraindications, and clinical context should be evaluated by healthcare professionals.
              </p>
            </div>
          </div>
        </div>
      ) : (
        <div className="bg-green-50 border-l-4 border-green-500 rounded-r-lg p-4 space-y-2">
          <div className="flex items-start gap-3">
            <svg className="w-5 h-5 text-green-600 flex-shrink-0 mt-0.5" fill="currentColor" viewBox="0 0 20 20">
              <path
                fillRule="evenodd"
                d="M10 18a8 8 0 100-16 8 8 0 000 16zm3.707-9.293a1 1 0 00-1.414-1.414L9 10.586 7.707 9.293a1 1 0 00-1.414 1.414l2 2a1 1 0 001.414 0l4-4z"
                clipRule="evenodd"
              />
            </svg>
            <div className="flex-1">
              <h4 className="text-sm font-semibold text-green-900 mb-1">✓ Strong Evidence Base - Standard of Care Guidance</h4>
              <p className="text-xs text-green-800 leading-relaxed">
                <strong>Very high verification level (L4)</strong> indicates strong, consistent evidence from high-quality sources (e.g., meta-analyses, systematic reviews, RCTs). This information aligns with established medical guidelines. However, healthcare decisions should always be individualized based on patient-specific factors and professional assessment.
              </p>
            </div>
          </div>
        </div>
      )}
      {/* Main Answer */}
      {parts[0] && (
        <p className={`text-[15.5px] font-medium tracking-tight leading-relaxed ${
          theme === 'dark' ? 'text-[#e3e3e3]' : 'text-gray-900'
        }`}>
          {parts[0].trim()}
        </p>
      )}

      {/* Evidence Summary Section */}
      {parts[1] === 'Evidence Summary' && parts[2] && (
        <div className="space-y-2">
          <p className={`text-[13px] font-semibold tracking-tight ${
            theme === 'dark' ? 'text-[#8e9196]' : 'text-gray-500'
          }`}>
            EVIDENCE SUMMARY
          </p>
          <div className={`text-[15px] font-medium leading-relaxed space-y-1.5 ${
            theme === 'dark' ? 'text-[#e3e3e3]' : 'text-gray-900'
          }`}>
            {parts[2].split('\n').filter(line => line.trim()).map((line, i) => {
              const trimmed = line.trim();
              if (trimmed.startsWith('-')) {
                return (
                  <div key={`evidence-${i}`} className="pl-4">
                    {trimmed.substring(1).trim()}
                  </div>
                );
              }
              return trimmed ? <div key={`evidence-${i}`}>{trimmed}</div> : null;
            })}
          </div>
        </div>
      )}

      {/* Limitations Section */}
      {parts.includes('Limitations') && (
        <div className="space-y-2">
          <p className={`text-[13px] font-semibold tracking-tight ${
            theme === 'dark' ? 'text-[#8e9196]' : 'text-gray-500'
          }`}>
            LIMITATIONS
          </p>
          <div className={`text-[15px] font-medium leading-relaxed space-y-1.5 ${
            theme === 'dark' ? 'text-[#e3e3e3]' : 'text-gray-900'
          }`}>
            {parts.at(-1)?.split('\n').filter(line => line.trim()).map((line, i) => {
              const trimmed = line.trim();
              if (trimmed.startsWith('-')) {
                return (
                  <div key={`limit-${i}`} className="pl-4">
                    {trimmed.substring(1).trim()}
                  </div>
                );
              }
              return trimmed ? <div key={`limit-${i}`}>{trimmed}</div> : null;
            })}
          </div>
        </div>
      )}
    </div>
  );
};
