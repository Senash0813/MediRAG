'use client';

import React from 'react';
import { LucideIcon } from 'lucide-react';
import { useTheme } from '@/contexts/ThemeContext';

interface ClusterCardProps {
  clusterNumber: number;
  name: string;
  icon: LucideIcon;
  description: string;
  onSelect: () => void;
  isSelected?: boolean;
}

export const ClusterCard = ({ name, icon: Icon, onSelect, isSelected = false }: ClusterCardProps) => {
  const { theme } = useTheme();

  return (
    <div
      onClick={onSelect}
      className={`
        border-orange-500/30 hover:border-orange-500/60
        ${theme === 'dark' ? 'bg-orange-500/5' : 'bg-white shadow-md hover:shadow-xl'}
        ${isSelected ? `ring-2 ring-orange-500 ring-offset-2 ${theme === 'dark' ? 'ring-offset-[#131314]' : 'ring-offset-[#d8dce5]'}` : ''}
        p-4 rounded-xl flex items-center gap-3 cursor-pointer
        transition-all border-2 group
        ${isSelected ? 'scale-[1.02]' : 'hover:scale-[1.02]'}
      `}
    >
      <div className={`w-8 h-8 rounded-lg flex items-center justify-center text-orange-400 ${
        theme === 'dark' ? 'bg-[#1e1f20]' : 'bg-gray-100'
      }`}>
        <Icon className="w-4 h-4" />
      </div>
      <h3 className={`text-base font-medium transition-colors ${
        theme === 'dark'
          ? 'text-[#e3e3e3] group-hover:text-white'
          : 'text-gray-900 group-hover:text-gray-800'
      }`}>
        {name}
      </h3>
    </div>
  );
};

