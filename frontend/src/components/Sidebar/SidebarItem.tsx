'use client';

import React from 'react';
import { useTheme } from '@/contexts/ThemeContext';

interface SidebarItemProps {
  icon: React.ReactNode;
  label: string;
  isOpen: boolean;
  onClick?: () => void;
}

export const SidebarItem = ({ icon, label, isOpen, onClick }: SidebarItemProps) => {
  const { theme } = useTheme();
  return (
    <button
      type="button"
      onClick={onClick}
      className={`flex items-center gap-3 p-3 rounded-full transition-colors ${!isOpen && 'justify-center'} ${
      theme === 'dark'
        ? 'hover:bg-[#282a2c] text-[#e3e3e3]'
        : 'hover:bg-gray-200 text-gray-700'
    }`}>
      {icon}
      {isOpen && <span className="text-sm font-medium">{label}</span>}
    </button>
  );
};

