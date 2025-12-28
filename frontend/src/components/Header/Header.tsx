'use client';

import React from 'react';
import { ChevronDown, UserCircle } from 'lucide-react';

export const Header = () => {
  return (
    <header className="flex items-center justify-between p-4 px-6">
      <div className="flex items-center gap-2 cursor-pointer group">
        <span className="text-xl font-medium">MediRAG</span>
        <ChevronDown className="w-4 h-4 text-[#8e9196] group-hover:text-white transition-colors" />
      </div>
      <div className="flex items-center gap-4">
        <div className="w-8 h-8 rounded-full bg-gradient-to-tr from-blue-500 to-purple-500 flex items-center justify-center cursor-pointer">
          <UserCircle className="w-6 h-6 text-white" />
        </div>
      </div>
    </header>
  );
};

