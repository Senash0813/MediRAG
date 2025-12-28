'use client';

import React, { useState } from 'react';
import { Sidebar } from '@/components/Sidebar/Sidebar';
import { Header } from '@/components/Header/Header';
import { ChatArea } from '@/components/ChatArea/ChatArea';
import { InputArea } from '@/components/InputArea/InputArea';

export default function Home() {
  const [isSidebarOpen, setIsSidebarOpen] = useState(true);

  return (
    <div className="flex h-screen bg-[#131314] text-[#e3e3e3] font-sans overflow-hidden">
      <Sidebar isOpen={isSidebarOpen} onToggle={() => setIsSidebarOpen(!isSidebarOpen)} />
      <main className="flex-1 flex flex-col relative overflow-hidden">
        <Header />
        <ChatArea />
        <InputArea />
      </main>
    </div>
  );
}
