'use client';

import React, { useState, useRef, useEffect } from 'react';
import { ChevronDown, UserCircle, Sparkles, Check } from 'lucide-react';

export const Header = () => {
  const [isDropdownOpen, setIsDropdownOpen] = useState(false);
  const dropdownRef = useRef<HTMLDivElement>(null);

  // Close dropdown when clicking outside
  useEffect(() => {
    const handleClickOutside = (event: MouseEvent) => {
      if (dropdownRef.current && !dropdownRef.current.contains(event.target as Node)) {
        setIsDropdownOpen(false);
      }
    };

    if (isDropdownOpen) {
      document.addEventListener('mousedown', handleClickOutside);
    }

    return () => {
      document.removeEventListener('mousedown', handleClickOutside);
    };
  }, [isDropdownOpen]);

  return (
    <header className="flex items-center justify-between p-4 px-6 relative">
      <div className="relative" ref={dropdownRef}>
        <div 
          className="flex items-center gap-2 cursor-pointer group"
          onClick={() => setIsDropdownOpen(!isDropdownOpen)}
        >
          <span className="text-xl font-medium">MediRAG v1</span>
          <ChevronDown className={`w-4 h-4 text-[#8e9196] group-hover:text-white transition-all ${isDropdownOpen ? 'rotate-180' : ''}`} />
        </div>

        {/* Dropdown Menu */}
        {isDropdownOpen && (
          <div className="absolute top-full left-0 mt-2 w-64 bg-[#1e1f20] border border-[#3c4043] rounded-xl shadow-xl z-50 overflow-hidden animate-in fade-in slide-in-from-top-2 duration-200 py-1">
            {/* Current Version */}
            <div className="py-2 px-3 mx-2 my-1 rounded-lg hover:bg-[#282a2c] transition-colors">
              <div className="flex items-center justify-between">
                <div className="flex-1">
                  <div className="flex items-center gap-2 mb-0.5">
                    <span className="text-base font-semibold text-[#e3e3e3]">MediRAG v1</span>
                  </div>
                  <p className="text-sm text-[#8e9196]">Current version</p>
                </div>
                <Check className="w-5 h-5 text-blue-400" />
              </div>
            </div>

            {/* Coming Soon Version */}
            <div className="py-2 px-3 mx-2 my-1 rounded-lg hover:bg-[#282a2c] transition-colors">
              <div className="flex items-center justify-between">
                <div className="flex-1">
                  <div className="flex items-center gap-2 mb-0.5">
                    <Sparkles className="w-4 h-4 text-blue-400" />
                    <span className="text-base font-semibold text-[#e3e3e3]">MediRAG v2</span>
                  </div>
                  <p className="text-sm text-[#8e9196]">Coming soon</p>
                </div>
              </div>
            </div>
          </div>
        )}
      </div>

      <div className="flex items-center gap-4">
        <div className="w-8 h-8 rounded-full bg-gradient-to-tr from-blue-500 to-purple-500 flex items-center justify-center cursor-pointer">
          <UserCircle className="w-6 h-6 text-white" />
        </div>
      </div>
    </header>
  );
};

