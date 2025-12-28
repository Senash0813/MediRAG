'use client';

import React from 'react';
import { Menu, Plus, History, Settings, HelpCircle } from 'lucide-react';
import { SidebarItem } from './SidebarItem';

interface SidebarProps {
  isOpen: boolean;
  onToggle: () => void;
}

export const Sidebar = ({ isOpen, onToggle }: SidebarProps) => {
  return (
    <aside 
      className={`${
        isOpen ? 'w-72' : 'w-20'
      } bg-[#1e1f20] flex flex-col transition-all duration-300 ease-in-out relative group`}
    >
      {/* Top Section */}
      <div className="p-4 flex flex-col gap-4">
        <button 
          onClick={onToggle}
          className="p-2 hover:bg-[#282a2c] rounded-full w-fit transition-colors"
        >
          <Menu className="w-6 h-6" />
        </button>
        
        <button className="flex items-center gap-3 bg-[#1a1c1e] hover:bg-[#282a2c] text-[#8e9196] py-3 px-4 rounded-full transition-colors mt-8 w-fit min-w-[56px]">
          <Plus className="w-5 h-5" />
          {isOpen && <span className="font-medium whitespace-nowrap">New chat</span>}
        </button>
      </div>

      {/* Bottom Section */}
      <div className="p-4 flex flex-col gap-1">
        <SidebarItem icon={<HelpCircle className="w-5 h-5" />} label="Help" isOpen={isOpen} />
        <SidebarItem icon={<History className="w-5 h-5" />} label="Activity" isOpen={isOpen} />
        <SidebarItem icon={<Settings className="w-5 h-5" />} label="Settings" isOpen={isOpen} />
        
        {isOpen && (
          <div className="mt-4 px-4 flex items-center gap-2 text-xs text-[#8e9196]">
            <div className="w-2 h-2 rounded-full bg-green-500"></div>
            <span>Colombo, Sri Lanka</span>
          </div>
        )}
      </div>
    </aside>
  );
};

