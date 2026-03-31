'use client';

import React, { useEffect, useState, useCallback } from 'react';
import { useTheme } from '@/contexts/ThemeContext';
import { X, Send } from 'lucide-react';

function DiscordIcon({ className }: { className?: string }) {
  return (
    <svg className={className} viewBox="0 0 24 24" fill="currentColor" aria-hidden>
      <path d="M20.317 4.37a19.791 19.791 0 0 0-4.885-1.515.074.074 0 0 0-.079.037c-.21.375-.444.864-.608 1.25a18.27 18.27 0 0 0-5.487 0 12.64 12.64 0 0 0-.617-1.25.077.077 0 0 0-.079-.037A19.736 19.736 0 0 0 3.677 4.37a.07.07 0 0 0-.032.027C.533 9.046-.32 13.58.099 18.057a.082.082 0 0 0 .031.057 19.9 19.9 0 0 0 5.993 3.03.078.078 0 0 0 .084-.028c.462-.63.874-1.295 1.226-1.994a.076.076 0 0 0-.041-.106 13.107 13.107 0 0 1-1.872-.892.077.077 0 0 1-.008-.128 10.2 10.2 0 0 0 .372-.292.074.074 0 0 1 .077-.01c3.928 1.793 8.18 1.793 12.062 0a.074.074 0 0 1 .078.01c.12.098.246.198.373.292a.077.077 0 0 1-.006.127 12.299 12.299 0 0 1-1.873.892.077.077 0 0 0-.041.107c.36.698.772 1.362 1.225 1.993a.076.076 0 0 0 .084.028 19.839 19.839 0 0 0 6.002-3.03.077.077 0 0 0 .032-.054c.5-5.177-.838-9.674-3.549-13.66a.061.061 0 0 0-.031-.03zM8.02 15.33c-1.183 0-2.157-1.085-2.157-2.419 0-1.333.956-2.419 2.157-2.419 1.21 0 2.176 1.096 2.157 2.42 0 1.333-.956 2.418-2.157 2.418zm7.975 0c-1.183 0-2.157-1.085-2.157-2.419 0-1.333.955-2.419 2.157-2.419 1.21 0 2.176 1.096 2.157 2.42 0 1.333-.946 2.418-2.157 2.418z" />
    </svg>
  );
}

function LinkedInIcon({ className }: { className?: string }) {
  return (
    <svg className={className} viewBox="0 0 24 24" fill="currentColor" aria-hidden>
      <path d="M20.447 20.452h-3.554v-5.569c0-1.328-.027-3.037-1.852-3.037-1.853 0-2.136 1.445-2.136 2.939v5.667H9.351V9h3.414v1.561h.046c.477-.9 1.637-1.85 3.37-1.85 3.601 0 4.267 2.37 4.267 5.455v6.286zM5.337 7.433c-1.144 0-2.063-.926-2.063-2.065 0-1.138.92-2.063 2.063-2.063 1.14 0 2.064.925 2.064 2.063 0 1.139-.925 2.065-2.064 2.065zm1.782 13.019H3.555V9h3.564v11.452zM22.225 0H1.771C.792 0 0 .774 0 1.729v20.542C0 23.227.792 24 1.771 24h20.451C23.2 24 24 23.227 24 22.271V1.729C24 .774 23.2 0 22.222 0h.003z" />
    </svg>
  );
}

/** Override in `.env.local`: NEXT_PUBLIC_HELP_EMAIL, NEXT_PUBLIC_DISCORD_URL, NEXT_PUBLIC_LINKEDIN_URL */
const HELP_EMAIL = process.env.NEXT_PUBLIC_HELP_EMAIL ?? 'support@medirag.example';
const DISCORD_URL = process.env.NEXT_PUBLIC_DISCORD_URL ?? 'https://discord.gg/your-invite';
const LINKEDIN_URL = process.env.NEXT_PUBLIC_LINKEDIN_URL ?? 'https://www.linkedin.com/company/your-company';

interface HelpModalProps {
  isOpen: boolean;
  onClose: () => void;
}

export function HelpModal({ isOpen, onClose }: HelpModalProps) {
  const { theme } = useTheme();
  const isDark = theme === 'dark';
  const [message, setMessage] = useState('');

  const resetMessage = useCallback(() => {
    setMessage('');
  }, []);

  const handleClose = useCallback(() => {
    resetMessage();
    onClose();
  }, [onClose, resetMessage]);

  useEffect(() => {
    if (isOpen) {
      document.body.style.overflow = 'hidden';
    } else {
      document.body.style.overflow = 'unset';
    }
    return () => {
      document.body.style.overflow = 'unset';
    };
  }, [isOpen]);

  useEffect(() => {
    const onEscape = (e: KeyboardEvent) => {
      if (e.key === 'Escape' && isOpen) handleClose();
    };
    window.addEventListener('keydown', onEscape);
    return () => window.removeEventListener('keydown', onEscape);
  }, [isOpen, handleClose]);

  if (!isOpen) return null;

  const handleBackdrop = (e: React.MouseEvent) => {
    if (e.target === e.currentTarget) handleClose();
  };

  const sendByEmail = () => {
    const trimmed = message.trim();
    if (!trimmed) return;
    const subject = encodeURIComponent('MediRAG — Help request');
    const body = encodeURIComponent(trimmed);
    window.location.href = `mailto:${HELP_EMAIL}?subject=${subject}&body=${body}`;
  };

  const sectionClass = isDark
    ? 'rounded-lg border border-gray-600 bg-gray-700/50 p-4'
    : 'rounded-lg border border-gray-200 bg-gray-50 p-4';

  const muted = isDark ? 'text-gray-400' : 'text-gray-500';

  const inputClass = `w-full min-h-[100px] resize-y rounded-lg border-2 px-3 py-2 text-sm outline-none box-border transition-[border-color] duration-150 ease-out ${
    isDark
      ? 'border-gray-600 bg-gray-800 text-white placeholder-gray-400 hover:border-gray-500 focus:border-blue-500 focus:outline-none'
      : 'border-gray-300 bg-white text-gray-900 placeholder-gray-500 hover:border-gray-400 focus:border-blue-500 focus:outline-none'
  }`;

  return (
    <div
      className="fixed inset-0 z-50 flex items-center justify-center p-4 animate-in fade-in duration-200"
      onClick={handleBackdrop}
    >
      <div className="absolute inset-0 bg-black/50 backdrop-blur-sm" />

      <div
        className={`relative w-full max-w-md max-h-[90vh] overflow-y-auto rounded-2xl shadow-2xl animate-in zoom-in-95 duration-200 border ${
          isDark ? 'bg-gray-800 border-gray-700' : 'bg-white border-gray-100'
        }`}
        role="dialog"
        aria-labelledby="help-title"
      >
        <button
          type="button"
          onClick={handleClose}
          className={`absolute top-4 right-4 p-2 rounded-full transition-colors ${
            isDark
              ? 'hover:bg-gray-700 text-gray-400 hover:text-white'
              : 'hover:bg-gray-100 text-gray-500 hover:text-gray-700'
          }`}
          aria-label="Close help"
        >
          <X className="w-5 h-5" />
        </button>

        <div className="p-6 sm:p-8">
          <h2
            id="help-title"
            className={`text-xl font-bold mb-1 text-center sm:text-2xl ${isDark ? 'text-gray-100' : 'text-gray-900'}`}
          >
            Help & community
          </h2>
          <p className={`text-sm text-center mb-5 ${muted}`}>
            Ask us a question or connect with us elsewhere.
          </p>

          <div className={sectionClass}>
            <label htmlFor="help-message" className={`block text-sm font-medium mb-2 ${isDark ? 'text-gray-200' : 'text-gray-700'}`}>
              Your question
            </label>
            <textarea
              id="help-message"
              value={message}
              onChange={(e) => setMessage(e.target.value)}
              className={inputClass}
              placeholder="Type your question here…"
              rows={4}
            />
            <button
              type="button"
              onClick={sendByEmail}
              disabled={!message.trim()}
              className="mt-3 w-full flex items-center justify-center gap-2 bg-linear-to-r from-blue-600 to-purple-600 hover:from-blue-700 hover:to-purple-700 disabled:opacity-50 disabled:cursor-not-allowed text-white font-semibold py-2.5 rounded-lg shadow-lg hover:shadow-xl text-sm transition-shadow duration-200"
            >
              <Send className="w-4 h-4" />
              Send via email
            </button>
            <p className={`mt-2 text-xs ${muted}`}>
              Opens your email app with your message addressed to our team.
            </p>
          </div>

          <div className="mt-4 flex flex-col gap-2">
            <a
              href={DISCORD_URL}
              target="_blank"
              rel="noopener noreferrer"
              className={`flex w-full items-center justify-center gap-2 px-6 py-2.5 border-2 rounded-lg font-medium transition-colors duration-200 shadow-sm hover:shadow text-sm ${
                isDark
                  ? 'bg-gray-700 border-gray-600 text-white hover:bg-gray-600'
                  : 'bg-white border-gray-300 text-gray-700 hover:bg-gray-50'
              }`}
            >
              <DiscordIcon className="h-4 w-4 shrink-0" />
              Join our Discord
            </a>
            <a
              href={LINKEDIN_URL}
              target="_blank"
              rel="noopener noreferrer"
              className={`flex w-full items-center justify-center gap-2 px-6 py-2.5 border-2 rounded-lg font-medium transition-colors duration-200 shadow-sm hover:shadow text-sm ${
                isDark
                  ? 'bg-gray-700 border-gray-600 text-white hover:bg-gray-600'
                  : 'bg-white border-gray-300 text-gray-700 hover:bg-gray-50'
              }`}
            >
              <LinkedInIcon className="h-4 w-4 shrink-0" />
              follow us on LinkedIn
            </a>
          </div>
        </div>
      </div>
    </div>
  );
}
