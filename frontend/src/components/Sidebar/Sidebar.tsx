'use client';

import React, { useState } from 'react';
import { Menu, Plus, History, Settings, HelpCircle } from 'lucide-react';
import { SidebarItem } from './SidebarItem';
import { SettingsModal } from './SettingsModal';
import { HelpModal } from './HelpModal';
import { useTheme } from '@/contexts/ThemeContext';

interface SidebarProps {
  isOpen: boolean;
  onToggle: () => void;
  onNewChat: () => void;
  onLoginClick?: () => void;
}

export const Sidebar = ({ isOpen, onToggle, onNewChat, onLoginClick }: SidebarProps) => {
  const { theme } = useTheme();
  const [settingsOpen, setSettingsOpen] = useState(false);
  const [helpOpen, setHelpOpen] = useState(false);
  return (
    <aside 
      className={`${
        isOpen ? 'w-72' : 'w-20'
      } flex flex-col transition-all duration-300 ease-in-out relative group ${
        theme === 'dark' ? 'bg-[#1e1f20]' : 'bg-gray-50'
      }`}
    >
      {/* Top Section */}
      <div className="p-4 flex flex-col gap-4">
        <button 
          onClick={onToggle}
          className={`p-2 rounded-full w-fit transition-colors ${
            theme === 'dark' 
              ? 'hover:bg-[#282a2c] text-[#e3e3e3]' 
              : 'hover:bg-gray-200 text-gray-700'
          }`}
        >
          <Menu className="w-6 h-6" />
        </button>
        
        <button 
          onClick={onNewChat}
          className={`flex items-center gap-3 py-3 px-4 rounded-full transition-colors mt-8 w-fit min-w-[56px] ${
            theme === 'dark'
              ? 'bg-[#1a1c1e] hover:bg-[#282a2c] text-[#8e9196]'
              : 'bg-gray-100 hover:bg-gray-200 text-gray-600'
          }`}
        >
          <Plus className="w-5 h-5" />
          {isOpen && <span className="font-medium whitespace-nowrap">New chat</span>}
        </button>
      </div>

      {/* Bottom Section */}
      <div className="p-4 flex flex-col gap-1">
        <SidebarItem
          icon={<HelpCircle className="w-5 h-5" />}
          label="Help"
          isOpen={isOpen}
          onClick={() => setHelpOpen(true)}
        />
        <SidebarItem icon={<History className="w-5 h-5" />} label="Activity" isOpen={isOpen} />
        <SidebarItem
          icon={<Settings className="w-5 h-5" />}
          label="Settings"
          isOpen={isOpen}
          onClick={() => setSettingsOpen(true)}
        />
      </div>

      <HelpModal isOpen={helpOpen} onClose={() => setHelpOpen(false)} />

      <SettingsModal
        isOpen={settingsOpen}
        onClose={() => setSettingsOpen(false)}
        onLoginClick={onLoginClick}
      />

      {/* Location - At the bottom */}
      {isOpen && (
        <div className={`mt-auto p-4 px-4 flex items-center gap-2 text-xs ${
          theme === 'dark' ? 'text-[#8e9196]' : 'text-gray-500'
        }`}>
          <div className="w-2 h-2 rounded-full bg-green-500"></div>
          <span>Colombo, Sri Lanka</span>
        </div>
      )}
    </aside>
  );
};

