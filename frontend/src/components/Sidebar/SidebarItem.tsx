'use client';

import React from 'react';

interface SidebarItemProps {
  icon: React.ReactNode;
  label: string;
  isOpen: boolean;
}

export const SidebarItem = ({ icon, label, isOpen }: SidebarItemProps) => (
  <button className={`flex items-center gap-3 p-3 hover:bg-[#282a2c] rounded-full text-[#e3e3e3] transition-colors ${!isOpen && 'justify-center'}`}>
    {icon}
    {isOpen && <span className="text-sm font-medium">{label}</span>}
  </button>
);

