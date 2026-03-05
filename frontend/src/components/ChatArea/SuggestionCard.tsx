'use client';

import React from 'react';

interface SuggestionCardProps {
  title: string;
  icon: React.ReactNode;
}

export const SuggestionCard = ({ title, icon }: SuggestionCardProps) => {
  return (
    <div 
      className="bg-[#1e1f20] hover:bg-[#282a2c] p-5 rounded-2xl h-48 flex flex-col justify-between cursor-pointer transition-all border border-transparent hover:border-[#3c4043] group"
    >
      <p className="text-base text-[#e3e3e3] leading-relaxed group-hover:text-white">
        {title}
      </p>
      <div className="self-end p-2 bg-[#131314] rounded-full group-hover:bg-[#1e1f20] transition-colors">
        {icon}
      </div>
    </div>
  );
};

