'use client';

import React, { useEffect } from 'react';
import Image from 'next/image';
import { useSession } from 'next-auth/react';
import { useTheme } from '@/contexts/ThemeContext';
import { X, User, Moon, Sun, Check, Building2 } from 'lucide-react';

interface SettingsModalProps {
  isOpen: boolean;
  onClose: () => void;
  onLoginClick?: () => void;
}

const individualFeatures = [
  'Full access to MediRAG Knowledge Base',
  'Priority response queue',
  '500 Credits per month',
];

export function SettingsModal({ isOpen, onClose, onLoginClick }: SettingsModalProps) {
  const { data: session, status } = useSession();
  const { theme, toggleTheme } = useTheme();
  const isDark = theme === 'dark';

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
      if (e.key === 'Escape' && isOpen) onClose();
    };
    window.addEventListener('keydown', onEscape);
    return () => window.removeEventListener('keydown', onEscape);
  }, [isOpen, onClose]);

  if (!isOpen) return null;

  const handleBackdrop = (e: React.MouseEvent) => {
    if (e.target === e.currentTarget) onClose();
  };

  /** Inner panels — same elevation language as LoginModal inputs / Google button */
  const sectionClass = isDark
    ? 'rounded-lg border border-gray-600 bg-gray-700/50 p-4'
    : 'rounded-lg border border-gray-200 bg-gray-50 p-4';

  const muted = isDark ? 'text-gray-400' : 'text-gray-500';
  const bodyText = isDark ? 'text-gray-200' : 'text-gray-700';

  return (
    <div
      className="fixed inset-0 z-50 flex items-center justify-center p-4 animate-in fade-in duration-200"
      onClick={handleBackdrop}
    >
      <div className="absolute inset-0 bg-black/50 backdrop-blur-sm" />

      <div
        className={`relative w-full max-w-lg max-h-[90vh] overflow-y-auto rounded-2xl shadow-2xl animate-in zoom-in-95 duration-200 border ${
          isDark ? 'bg-gray-800 border-gray-700' : 'bg-white border-gray-100'
        }`}
        role="dialog"
        aria-labelledby="settings-title"
      >
        <button
          type="button"
          onClick={onClose}
          className={`absolute top-4 right-4 p-2 rounded-full transition-colors ${
            isDark
              ? 'hover:bg-gray-700 text-gray-400 hover:text-white'
              : 'hover:bg-gray-100 text-gray-500 hover:text-gray-700'
          }`}
          aria-label="Close settings"
        >
          <X className="w-5 h-5" />
        </button>

        <div className="p-8">
          <h2
            id="settings-title"
            className={`text-2xl font-bold mb-2 text-center ${isDark ? 'text-gray-100' : 'text-gray-900'}`}
          >
            Account & subscription
          </h2>
          <p className={`text-sm text-center mb-6 ${isDark ? 'text-gray-400' : 'text-gray-500'}`}>
            Manage your profile, appearance, and plan options.
          </p>

          {/* Account */}
          <div className={`mb-4 ${sectionClass}`}>
            <h3 className={`text-sm font-medium mb-1 ${bodyText}`}>Account</h3>
            <p className={`text-xs ${muted}`}>Your sign-in details.</p>
            {status === 'loading' ? (
              <div className={`mt-4 h-16 animate-pulse rounded-lg ${isDark ? 'bg-gray-800' : 'bg-gray-200'}`} />
            ) : session?.user ? (
              <div className="mt-4 flex items-center gap-3">
                {session.user.image ? (
                  <Image
                    src={session.user.image}
                    alt=""
                    width={48}
                    height={48}
                    className="h-12 w-12 rounded-full object-cover"
                    unoptimized
                  />
                ) : (
                  <div className="flex h-12 w-12 items-center justify-center rounded-full bg-linear-to-tr from-blue-500 to-purple-500">
                    <User className="h-6 w-6 text-white" />
                  </div>
                )}
                <div className="min-w-0 flex-1">
                  <p className={`truncate text-sm font-medium ${isDark ? 'text-gray-100' : 'text-gray-900'}`}>
                    {session.user.name || 'User'}
                  </p>
                  <p className={`truncate text-xs ${muted}`}>{session.user.email}</p>
                </div>
              </div>
            ) : (
              <div className="mt-4">
                <p className={`text-sm ${muted}`}>
                  Sign in to sync your account and access subscription options.
                </p>
                {onLoginClick && (
                  <button
                    type="button"
                    onClick={() => {
                      onClose();
                      onLoginClick();
                    }}
                    className="mt-3 w-full bg-linear-to-r from-blue-600 to-purple-600 hover:from-blue-700 hover:to-purple-700 text-white font-semibold py-3 rounded-lg transition-all duration-200 shadow-lg hover:shadow-xl"
                  >
                    Sign in
                  </button>
                )}
              </div>
            )}
          </div>

          {/* Appearance */}
          <div className={`mb-4 ${sectionClass}`}>
            <h3 className={`text-sm font-medium mb-1 ${bodyText}`}>Appearance</h3>
            <p className={`text-xs ${muted}`}>Choose light or dark theme for the app.</p>
            <div className="mt-4 flex flex-col gap-3 sm:flex-row sm:items-center sm:justify-between">
              <span className={`text-sm font-medium ${isDark ? 'text-gray-200' : 'text-gray-700'}`}>
                {isDark ? 'Dark mode' : 'Light mode'}
              </span>
              <button
                type="button"
                onClick={toggleTheme}
                className={`flex items-center justify-center gap-2 px-6 py-3 border-2 rounded-lg font-medium transition-colors duration-200 shadow-sm hover:shadow ${
                  isDark
                    ? 'bg-gray-700 border-gray-600 text-white hover:bg-gray-600'
                    : 'bg-white border-gray-300 text-gray-700 hover:bg-gray-50'
                }`}
                title={isDark ? 'Switch to light mode' : 'Switch to dark mode'}
              >
                {isDark ? <Moon className="h-5 w-5" /> : <Sun className="h-5 w-5" />}
                Toggle theme
              </button>
            </div>
          </div>

          {/* Subscription */}
          <div className="space-y-3">
            <h3 className={`text-sm font-medium ${bodyText}`}>Subscription</h3>
            <div className="grid gap-4 sm:grid-cols-2">
              <div className={sectionClass}>
                <div className="flex items-start justify-between gap-2">
                  <div>
                    <p className={`text-sm font-semibold ${isDark ? 'text-gray-100' : 'text-gray-900'}`}>Individual</p>
                    <p className={`text-xs ${muted}`}>Single user</p>
                  </div>
                  <span className={`text-lg font-bold ${isDark ? 'text-gray-100' : 'text-gray-900'}`}>$25</span>
                </div>
                <p className={`mt-1 text-xs ${muted}`}>per month</p>
                <ul className="mt-3 space-y-2">
                  {individualFeatures.map((f) => (
                    <li key={f} className={`flex items-start gap-2 text-sm ${isDark ? 'text-gray-300' : 'text-gray-600'}`}>
                      <span
                        className={`mt-0.5 flex h-5 w-5 shrink-0 items-center justify-center rounded-full border bg-blue-500/10 text-blue-500 ${isDark ? 'border-gray-600' : 'border-gray-300'}`}
                      >
                        <Check className="h-3 w-3" strokeWidth={3} />
                      </span>
                      {f}
                    </li>
                  ))}
                </ul>
                <button
                  type="button"
                  disabled
                  className={`mt-4 w-full py-3 rounded-lg text-sm font-semibold transition-colors disabled:opacity-50 disabled:cursor-not-allowed border-2 ${
                    isDark
                      ? 'border-gray-600 text-gray-400 bg-gray-800/50'
                      : 'border-gray-200 text-gray-400 bg-gray-50'
                  }`}
                >
                  Coming soon
                </button>
              </div>

              <div className={sectionClass}>
                <div className="flex items-start gap-2">
                  <Building2 className={`h-5 w-5 shrink-0 ${muted}`} />
                  <div>
                    <p className={`text-sm font-semibold ${isDark ? 'text-gray-100' : 'text-gray-900'}`}>
                      Institutional
                    </p>
                    <p className={`mt-1 text-xs ${muted}`}>
                      Volume access with a shared quota for your organization. Get a custom quote.
                    </p>
                  </div>
                </div>
                <a
                  href="mailto:support@medirag.example?subject=Institutional%20access%20inquiry"
                  className={`mt-4 flex w-full items-center justify-center px-6 py-3 border-2 rounded-lg font-medium transition-colors duration-200 shadow-sm hover:shadow text-sm ${
                    isDark
                      ? 'bg-gray-700 border-gray-600 text-white hover:bg-gray-600'
                      : 'bg-white border-gray-300 text-gray-700 hover:bg-gray-50'
                  }`}
                >
                  Get a quote
                </a>
              </div>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}
