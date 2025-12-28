'use client';

import React from 'react';
import { Image as ImageIcon, Mic, Send } from 'lucide-react';

export const InputArea = () => {
  return (
    <div className="absolute bottom-0 left-0 w-full p-4 md:p-6 bg-gradient-to-t from-[#131314] via-[#131314] to-transparent">
      <div className="max-w-4xl mx-auto w-full flex flex-col gap-3">
        <div className="relative group">
          <div className="flex items-center bg-[#1e1f20] rounded-[28px] px-6 py-4 border border-transparent focus-within:border-[#3c4043] shadow-lg transition-all">
            <input 
              type="text" 
              placeholder="Ask your question"
              className="flex-1 bg-transparent border-none outline-none text-lg text-[#e3e3e3] placeholder-[#8e9196] py-1"
            />
            <div className="flex items-center gap-2 md:gap-4 ml-4">
              {/* <button className="p-2 hover:bg-[#282a2c] rounded-full transition-colors text-[#e3e3e3]">
                <ImageIcon className="w-6 h-6" />
              </button> */}
              <button className="p-2 hover:bg-[#282a2c] rounded-full transition-colors text-[#e3e3e3]">
                <Mic className="w-6 h-6" />
              </button>
              <button className="p-3 bg-[#282a2c] rounded-full text-[#8e9196] hover:text-white transition-all">
                <Send className="w-5 h-5" />
              </button>
            </div>
          </div>
        </div>
        <p className="text-center text-xs text-[#8e9196] px-4 leading-relaxed">
          Answers are grounded in MIRIAD medical literature. Always verify important information with a professional.
        </p>
      </div>
    </div>
  );
};

