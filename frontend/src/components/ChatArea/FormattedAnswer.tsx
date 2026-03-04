'use client';

import React from 'react';
import { useTheme } from '@/contexts/ThemeContext';

interface FormattedAnswerProps {
  answer: string;
  isCluster4?: boolean;
}

export const FormattedAnswer = ({ answer, isCluster4 = false }: FormattedAnswerProps) => {
  const { theme } = useTheme();

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
