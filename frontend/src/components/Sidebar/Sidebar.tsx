'use client';

import React, { useState, useEffect, useCallback } from 'react';
import { Menu, Plus, History, Settings, HelpCircle, MessageSquare } from 'lucide-react';
import { useSession } from 'next-auth/react';
import { SidebarItem } from './SidebarItem';
import { SettingsModal } from './SettingsModal';
import { HelpModal } from './HelpModal';
import { useTheme } from '@/contexts/ThemeContext';
import type { ConversationListItem } from '@/types/conversation';

interface SidebarProps {
  isOpen: boolean;
  onToggle: () => void;
  onNewChat: () => void;
  onLoginClick?: () => void;
  /** Bumps when server-side chat history changes; refreshes the list. */
  historyVersion?: number;
  activeConversationId?: string | null;
  onOpenConversation?: (id: string) => void | Promise<void>;
}

export const Sidebar = ({
  isOpen,
  onToggle,
  onNewChat,
  onLoginClick,
  historyVersion = 0,
  activeConversationId = null,
  onOpenConversation,
}: SidebarProps) => {
  const { theme } = useTheme();
  const { status } = useSession();
  const [settingsOpen, setSettingsOpen] = useState(false);
  const [helpOpen, setHelpOpen] = useState(false);
  const [conversations, setConversations] = useState<ConversationListItem[]>([]);
  const [listLoading, setListLoading] = useState(false);

  const fetchConversations = useCallback(async () => {
    if (status === 'loading') return;
    if (status !== 'authenticated') {
      setConversations([]);
      return;
    }
    setListLoading(true);
    try {
      const r = await fetch('/api/conversations');
      if (!r.ok) return;
      const data = (await r.json()) as { conversations?: ConversationListItem[] };
      setConversations(data.conversations ?? []);
    } catch {
      setConversations([]);
    } finally {
      setListLoading(false);
    }
  }, [status]);

  useEffect(() => {
    void fetchConversations();
  }, [fetchConversations, historyVersion]);

  const showHistory = status === 'authenticated' && isOpen;

  return (
    <aside
      className={`${
        isOpen ? 'w-72' : 'w-20'
      } flex flex-col h-full min-h-0 transition-all duration-300 ease-in-out relative group ${
        theme === 'dark' ? 'bg-[#1e1f20]' : 'bg-gray-50'
      }`}
    >
      {/* Top Section */}
      <div className="p-4 flex flex-col gap-4 shrink-0">
        <button
          type="button"
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
          type="button"
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

      {/* Chat history (signed-in only, expanded sidebar) */}
      {showHistory && (
        <div className="flex-1 min-h-0 flex flex-col px-3 pb-2">
          <p
            className={`text-xs font-medium uppercase tracking-wide px-3 py-2 shrink-0 ${
              theme === 'dark' ? 'text-[#8e9196]' : 'text-gray-500'
            }`}
          >
            Recent
          </p>
          <div className="flex-1 min-h-0 overflow-y-auto rounded-lg pr-1 -mr-1">
            {listLoading && conversations.length === 0 ? (
              <p
                className={`px-3 py-2 text-sm ${
                  theme === 'dark' ? 'text-[#8e9196]' : 'text-gray-500'
                }`}
              >
                Loading…
              </p>
            ) : conversations.length === 0 ? (
              <p
                className={`px-3 py-2 text-sm ${
                  theme === 'dark' ? 'text-[#8e9196]' : 'text-gray-500'
                }`}
              >
                No chats yet. Start one with New chat.
              </p>
            ) : (
              <ul className="flex flex-col gap-0.5">
                {conversations.map((c) => {
                  const isActive = activeConversationId === c.id;
                  return (
                    <li key={c.id}>
                      <button
                        type="button"
                        onClick={() => void onOpenConversation?.(c.id)}
                        className={`w-full text-left flex items-start gap-2 rounded-xl px-3 py-2.5 transition-colors ${
                          isActive
                            ? theme === 'dark'
                              ? 'bg-[#282a2c] text-[#e3e3e3]'
                              : 'bg-gray-200 text-gray-900'
                            : theme === 'dark'
                              ? 'hover:bg-[#282a2c]/70 text-[#e3e3e3]'
                              : 'hover:bg-gray-200/80 text-gray-800'
                        }`}
                      >
                        <MessageSquare
                          className={`w-4 h-4 shrink-0 mt-0.5 ${
                            theme === 'dark' ? 'text-[#8e9196]' : 'text-gray-500'
                          }`}
                        />
                        <span className="min-w-0 flex-1">
                          <span className="block text-sm font-medium leading-snug truncate">
                            {c.title || 'New chat'}
                          </span>
                          <span
                            className={`block text-xs mt-0.5 truncate ${
                              theme === 'dark' ? 'text-[#8e9196]' : 'text-gray-500'
                            }`}
                          >
                            {new Date(c.updatedAt).toLocaleString(undefined, {
                              month: 'short',
                              day: 'numeric',
                              hour: 'numeric',
                              minute: '2-digit',
                            })}
                            {c.messageCount > 0 ? ` · ${c.messageCount} msgs` : ''}
                          </span>
                        </span>
                      </button>
                    </li>
                  );
                })}
              </ul>
            )}
          </div>
        </div>
      )}

      {/* Bottom Section */}
      <div className="p-4 flex flex-col gap-1 shrink-0">
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

      {isOpen && (
        <div
          className={`shrink-0 p-4 px-4 flex items-center gap-2 text-xs ${
            theme === 'dark' ? 'text-[#8e9196]' : 'text-gray-500'
          }`}
        >
          <div className="w-2 h-2 rounded-full bg-green-500" />
          <span>Colombo, Sri Lanka</span>
        </div>
      )}
    </aside>
  );
};
