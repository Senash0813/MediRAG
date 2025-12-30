'use client';

import React from 'react';
import { LucideIcon } from 'lucide-react';

interface ClusterCardProps {
  clusterNumber: number;
  name: string;
  icon: LucideIcon;
  description: string;
  onSelect: () => void;
  isSelected?: boolean;
}

export const ClusterCard = ({ clusterNumber, name, icon: Icon, description, onSelect, isSelected = false }: ClusterCardProps) => {
  const colorClasses = [
    'border-blue-500/30 hover:border-blue-500/60 bg-blue-500/5',
    'border-purple-500/30 hover:border-purple-500/60 bg-purple-500/5',
    'border-green-500/30 hover:border-green-500/60 bg-green-500/5',
    'border-orange-500/30 hover:border-orange-500/60 bg-orange-500/5',
  ];

  const iconColors = [
    'text-blue-400',
    'text-purple-400',
    'text-green-400',
    'text-orange-400',
  ];

  const colorClass = colorClasses[clusterNumber - 1];
  const iconColor = iconColors[clusterNumber - 1];

  return (
    <div 
      onClick={onSelect}
      className={`
        ${colorClass}
        ${isSelected ? 'ring-2 ring-offset-2 ring-offset-[#131314]' : ''}
        p-4 rounded-xl flex flex-col cursor-pointer 
        transition-all border-2 group
        ${isSelected ? 'scale-[1.02]' : 'hover:scale-[1.02]'}
      `}
    >
      <div className="flex items-center gap-3 mb-4">
        <div className={`w-8 h-8 rounded-lg bg-[#1e1f20] flex items-center justify-center ${iconColor}`}>
          <Icon className="w-4 h-4" />
        </div>
        <h3 className="text-base font-medium text-[#e3e3e3] group-hover:text-white transition-colors">
          {name}
        </h3>
      </div>
      
      <div className="flex-1">
        <p className="text-[12.5px] font-medium text-[#e3e3e3] group-hover:text-white transition-colors leading-relaxed">
          {description}
        </p>
      </div>
    </div>
  );
};

