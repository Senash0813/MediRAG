'use client';

import React, { useState, KeyboardEvent } from 'react';
import { Image as ImageIcon, Mic, Send, Loader2 } from 'lucide-react';

interface InputAreaProps {
  onSend: (question: string) => void;
  isLoading: boolean;
  disabled?: boolean;
  selectedClusterName?: string | null;
  hasMessages: boolean;
}

export const InputArea = ({ onSend, isLoading, disabled = false, selectedClusterName, hasMessages }: InputAreaProps) => {
  const [input, setInput] = useState('');

  const handleSend = () => {
    if (input.trim() && !isLoading && !disabled) {
      onSend(input.trim());
      setInput('');
    }
  };

  const handleKeyPress = (e: KeyboardEvent<HTMLInputElement>) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      handleSend();
    }
  };

  return (
    <div className={`absolute bottom-0 left-0 w-full bg-gradient-to-t from-[#131314] via-[#131314] to-transparent transition-all duration-500 ${
      hasMessages ? 'p-4 md:p-6' : 'p-4 md:p-6'
    }`}>
      <div className={`max-w-4xl mx-auto w-full flex flex-col gap-3 transition-all duration-500 ${
        hasMessages ? 'max-w-2xl' : ''
      }`}>
        <div className="relative group">
          <div className={`flex items-center bg-[#1e1f20] rounded-[28px] px-6 py-4 border border-transparent focus-within:border-[#3c4043] shadow-lg transition-all duration-300 ${
            hasMessages ? 'shadow-xl' : ''
          }`}>
            <input 
              type="text" 
              placeholder={
                disabled 
                  ? "Select a cluster first" 
                  : selectedClusterName 
                    ? `Ask a question about ${selectedClusterName}...` 
                    : "Ask your question"
              }
              value={input}
              onChange={(e) => setInput(e.target.value)}
              onKeyPress={handleKeyPress}
              disabled={disabled || isLoading}
              className="flex-1 bg-transparent border-none outline-none text-lg text-[#e3e3e3] placeholder-[#8e9196] py-1 disabled:opacity-50"
            />
            <div className="flex items-center gap-2 md:gap-4 ml-4">
              {/* <button className="p-2 hover:bg-[#282a2c] rounded-full transition-colors text-[#e3e3e3]">
                <ImageIcon className="w-6 h-6" />
              </button> */}
              <button 
                className="p-2 hover:bg-[#282a2c] rounded-full transition-colors text-[#e3e3e3] disabled:opacity-50"
                disabled={disabled || isLoading}
              >
                <Mic className="w-6 h-6" />
              </button>
              <button 
                onClick={handleSend}
                disabled={!input.trim() || isLoading || disabled}
                className="p-3 bg-[#282a2c] rounded-full text-[#8e9196] hover:text-white transition-all disabled:opacity-50 disabled:cursor-not-allowed"
              >
                {isLoading ? (
                  <Loader2 className="w-5 h-5 animate-spin" />
                ) : (
                  <Send className="w-5 h-5" />
                )}
              </button>
            </div>
          </div>
        </div>
        {!hasMessages && (
          <p className="text-center text-xs text-[#8e9196] px-4 leading-relaxed animate-in fade-in duration-300">
            Answers are grounded in MIRIAD medical literature. Always verify important information with a professional.
          </p>
        )}
      </div>
    </div>
  );
};

