'use client';

import React from 'react';
import { X } from 'lucide-react';

interface SelectedClusterDetailsProps {
  clusterNumber: number;
  clusterName: string;
  specialties: string[];
  onDeselect: () => void;
}

export const SelectedClusterDetails = ({ 
  clusterNumber, 
  clusterName, 
  specialties, 
  onDeselect 
}: SelectedClusterDetailsProps) => {
  const colorClasses = [
    'border-blue-500/30 bg-blue-500/5',
    'border-purple-500/30 bg-purple-500/5',
    'border-green-500/30 bg-green-500/5',
    'border-orange-500/30 bg-orange-500/5',
  ];

  const colorClass = colorClasses[clusterNumber - 1];

  return (
    <div className={`mt-6 p-5 rounded-xl border-2 ${colorClass} animate-in fade-in slide-in-from-top-4 duration-500`}>
      <div className="flex items-center justify-between mb-4">
        <div>
          <h3 className="text-lg font-semibold text-[#e3e3e3] mb-1">
            {clusterName} - Pipeline {clusterNumber}
          </h3>
          <p className="text-xs text-[#8e9196]">Selected cluster specialties</p>
        </div>
        <button
          onClick={onDeselect}
          className="p-2 hover:bg-[#282a2c] rounded-lg transition-colors text-[#8e9196] hover:text-white"
          aria-label="Deselect cluster"
        >
          <X className="w-4 h-4" />
        </button>
      </div>
      
      <div className="flex flex-wrap gap-2">
        {specialties.map((specialty, idx) => (
          <span
            key={idx}
            className="px-3 py-1.5 bg-[#1e1f20] border border-[#3c4043] rounded-lg text-xs text-[#e3e3e3]"
          >
            {specialty}
          </span>
        ))}
      </div>
    </div>
  );
};

