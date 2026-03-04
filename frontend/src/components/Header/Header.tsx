'use client';

import React, { useState, useRef, useEffect } from 'react';
import { ChevronDown, UserCircle, Sparkles, Check, LogOut, User, LogIn } from 'lucide-react';
import { useTheme } from '@/contexts/ThemeContext';
import { useSession, signOut } from 'next-auth/react';

interface HeaderProps {
  onLoginClick?: () => void;
}

export const Header = ({ onLoginClick }: HeaderProps) => {
  const { theme } = useTheme();
  const { data: session, status } = useSession();
  const [isDropdownOpen, setIsDropdownOpen] = useState(false);
  const [isUserMenuOpen, setIsUserMenuOpen] = useState(false);
  const dropdownRef = useRef<HTMLDivElement>(null);
  const userMenuRef = useRef<HTMLDivElement>(null);

  // Close dropdown when clicking outside
  useEffect(() => {
    const handleClickOutside = (event: MouseEvent) => {
      if (dropdownRef.current && !dropdownRef.current.contains(event.target as Node)) {
        setIsDropdownOpen(false);
      }
      if (userMenuRef.current && !userMenuRef.current.contains(event.target as Node)) {
        setIsUserMenuOpen(false);
      }
    };

    if (isDropdownOpen || isUserMenuOpen) {
      document.addEventListener('mousedown', handleClickOutside);
    }

    return () => {
      document.removeEventListener('mousedown', handleClickOutside);
    };
  }, [isDropdownOpen, isUserMenuOpen]);

  const handleLogout = async () => {
    await signOut({ callbackUrl: '/' });
  };

  return (
    <header className="flex items-center justify-between p-4 px-6 relative">
      <div className="relative" ref={dropdownRef}>
        <div 
          className="flex items-center gap-2 cursor-pointer group"
          onClick={() => setIsDropdownOpen(!isDropdownOpen)}
          role="button"
          tabIndex={0}
          onKeyDown={(e) => {
            if (e.key === 'Enter' || e.key === ' ') {
              setIsDropdownOpen(!isDropdownOpen);
            }
          }}
        >
          <span className={`text-xl font-medium ${theme === 'dark' ? 'text-[#e3e3e3]' : 'text-gray-900'}`}>MediRAG v1</span>
          <ChevronDown className={`w-4 h-4 transition-all ${isDropdownOpen ? 'rotate-180' : ''} ${
            theme === 'dark' ? 'text-[#8e9196] group-hover:text-white' : 'text-gray-500 group-hover:text-gray-700'
          }`} />
        </div>

        {/* Dropdown Menu */}
        {isDropdownOpen && (
          <div className={`absolute top-full left-0 mt-2 w-64 rounded-xl shadow-xl z-50 overflow-hidden animate-in fade-in slide-in-from-top-2 duration-200 py-1 ${
            theme === 'dark'
              ? 'bg-[#1e1f20] border border-[#3c4043]'
              : 'bg-white border border-gray-200'
          }`}>
            {/* Current Version */}
            <div className={`py-2 px-3 mx-2 my-1 rounded-lg transition-colors ${
              theme === 'dark' ? 'hover:bg-[#282a2c]' : 'hover:bg-gray-100'
            }`}>
              <div className="flex items-center justify-between">
                <div className="flex-1">
                  <div className="flex items-center gap-2 mb-0.5">
                    <span className={`text-base font-semibold ${
                      theme === 'dark' ? 'text-[#e3e3e3]' : 'text-gray-900'
                    }`}>MediRAG v1</span>
                  </div>
                  <p className={`text-sm ${
                    theme === 'dark' ? 'text-[#8e9196]' : 'text-gray-500'
                  }`}>Current version</p>
                </div>
                <Check className="w-5 h-5 text-blue-400" />
              </div>
            </div>

            {/* Coming Soon Version */}
            <div className={`py-2 px-3 mx-2 my-1 rounded-lg transition-colors ${
              theme === 'dark' ? 'hover:bg-[#282a2c]' : 'hover:bg-gray-100'
            }`}>
              <div className="flex items-center justify-between">
                <div className="flex-1">
                  <div className="flex items-center gap-2 mb-0.5">
                    <Sparkles className="w-4 h-4 text-blue-400" />
                    <span className={`text-base font-semibold ${
                      theme === 'dark' ? 'text-[#e3e3e3]' : 'text-gray-900'
                    }`}>MediRAG v2</span>
                  </div>
                  <p className={`text-sm ${
                    theme === 'dark' ? 'text-[#8e9196]' : 'text-gray-500'
                  }`}>Coming soon</p>
                </div>
              </div>
            </div>
          </div>
        )}
      </div>

      {/* User Menu / Login Button */}
      <div className="flex items-center gap-4">
        {status === 'loading' ? (
          <div className="w-8 h-8 rounded-full bg-gray-300 dark:bg-gray-700 animate-pulse" />
        ) : session?.user ? (
          // Logged in user menu
          <div className="relative" ref={userMenuRef}>
            <div
              className="flex items-center gap-2 cursor-pointer group"
              onClick={() => setIsUserMenuOpen(!isUserMenuOpen)}
              role="button"
              tabIndex={0}
              onKeyDown={(e) => {
                if (e.key === 'Enter' || e.key === ' ') {
                  setIsUserMenuOpen(!isUserMenuOpen);
                }
              }}
            >
              {session.user.image ? (
                <img 
                  src={session.user.image} 
                  alt={session.user.name || 'User'}
                  className="w-8 h-8 rounded-full"
                />
              ) : (
                <div className="w-8 h-8 rounded-full bg-gradient-to-tr from-blue-500 to-purple-500 flex items-center justify-center">
                  <UserCircle className="w-6 h-6 text-white" />
                </div>
              )}
              <ChevronDown className={`w-4 h-4 transition-all ${isUserMenuOpen ? 'rotate-180' : ''} ${
                theme === 'dark' ? 'text-[#8e9196] group-hover:text-white' : 'text-gray-500 group-hover:text-gray-700'
              }`} />
            </div>

            {/* User Dropdown */}
            {isUserMenuOpen && (
              <div className={`absolute top-full right-0 mt-2 w-64 rounded-xl shadow-xl z-50 overflow-hidden animate-in fade-in slide-in-from-top-2 duration-200 py-2 ${
                theme === 'dark'
                  ? 'bg-[#1e1f20] border border-[#3c4043]'
                  : 'bg-white border border-gray-200'
              }`}>
                {/* User Info */}
                <div className={`py-3 px-4 border-b ${
                  theme === 'dark' ? 'border-[#3c4043]' : 'border-gray-200'
                }`}>
                  <div className="flex items-center gap-3">
                    {session.user.image ? (
                      <img 
                        src={session.user.image} 
                        alt={session.user.name || 'User'}
                        className="w-10 h-10 rounded-full"
                      />
                    ) : (
                      <div className="w-10 h-10 rounded-full bg-gradient-to-tr from-blue-500 to-purple-500 flex items-center justify-center">
                        <User className="w-6 h-6 text-white" />
                      </div>
                    )}
                    <div className="flex-1 min-w-0">
                      <p className={`text-sm font-medium truncate ${
                        theme === 'dark' ? 'text-[#e3e3e3]' : 'text-gray-900'
                      }`}>
                        {session.user.name || 'User'}
                      </p>
                      <p className={`text-xs truncate ${
                        theme === 'dark' ? 'text-[#8e9196]' : 'text-gray-500'
                      }`}>
                        {session.user.email}
                      </p>
                    </div>
                  </div>
                </div>

                {/* Logout Button */}
                <button
                  onClick={handleLogout}
                  className={`w-full flex items-center gap-3 py-2 px-4 text-left transition-colors ${
                    theme === 'dark' 
                      ? 'hover:bg-[#282a2c] text-[#e3e3e3]' 
                      : 'hover:bg-gray-100 text-gray-900'
                  }`}
                >
                  <LogOut className="w-4 h-4" />
                  <span className="text-sm">Sign out</span>
                </button>
              </div>
            )}
          </div>
        ) : (
          // Guest user - Login button
          <button
            onClick={onLoginClick}
            className={`flex items-center gap-2 px-4 py-2 rounded-lg font-medium transition-all duration-200 ${
              theme === 'dark'
                ? 'bg-blue-600 hover:bg-blue-700 text-white'
                : 'bg-blue-600 hover:bg-blue-700 text-white'
            } shadow-lg hover:shadow-xl`}
          >
            <LogIn className="w-4 h-4" />
            <span>Login</span>
          </button>
        )}
      </div>
    </header>
  );
};

