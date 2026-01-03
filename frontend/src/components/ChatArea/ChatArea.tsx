'use client';

import React, { useState, useEffect, useRef } from 'react';
import { Brain, Heart, Stethoscope, Activity, Sparkles, Copy, Check, Moon, Sun } from 'lucide-react';
import { ClusterCard } from './ClusterCard';
import { useTheme } from '@/contexts/ThemeContext';

interface Message {
  question: string;
  answer: string;
  timestamp: Date;
}

interface ChatAreaProps {
  selectedCluster: number | null;
  messages: Message[];
  onClusterSelect: (clusterNumber: number) => void;
  pendingQuestion: string | null;
  isLoading: boolean;
}

// Rotating loading messages
const loadingMessages = [
  'Generating answer...',
  'Analyzing question...',
  'Searching knowledge base...',
  'Preparing response...',
];

export const ChatArea = ({ selectedCluster, messages, onClusterSelect, pendingQuestion, isLoading }: ChatAreaProps) => {
  const [currentLoadingMessage, setCurrentLoadingMessage] = useState(0);
  const startTimeRef = useRef<number | null>(null);
  const { theme, toggleTheme } = useTheme();
  const [copiedIndex, setCopiedIndex] = useState<{ type: 'question' | 'answer'; index: number } | null>(null);

  const handleCopy = async (text: string, type: 'question' | 'answer', index: number) => {
    try {
      await navigator.clipboard.writeText(text);
      setCopiedIndex({ type, index });
      setTimeout(() => setCopiedIndex(null), 2000);
    } catch (err) {
      console.error('Failed to copy:', err);
    }
  };
  
  // Rotate through loading messages
  useEffect(() => {
    if (isLoading && pendingQuestion) {
      // Start timer when loading begins
      startTimeRef.current = Date.now();
      
      const interval = setInterval(() => {
        if (startTimeRef.current !== null) {
          const elapsed = Math.floor((Date.now() - startTimeRef.current) / 2000);
          setCurrentLoadingMessage(elapsed % loadingMessages.length);
        }
      }, 2000); // Update every 2 seconds
      
      return () => {
        clearInterval(interval);
        startTimeRef.current = null;
      };
    }
  }, [isLoading, pendingQuestion]);
  const clusters = [
    {
      number: 1,
      name: 'Neurosciences',
      icon: Brain,
      description: 'Specialized care for disorders of the brain, spinal cord, and nervous system.',
      specialties: [
        'Neurology',
        'Neurosurgery',
      ],
    },
    {
      number: 2,
      name: 'Cardiovascular',
      icon: Heart,
      description: 'Comprehensive heart and vascular system care, including surgical interventions.',
      specialties: [
        'Cardiology',
        'Cardiothoracic Surgery',
        'Vascular Surgery',
      ],
    },
    {
      number: 3,
      name: 'Internal Medicine',
      icon: Stethoscope,
      description: 'Broad range of medical specialties covering organ systems and general health.',
      specialties: [
        'General Internal Medicine',
        'General Pediatrics',
        'General Surgery',
        'Nephrology',
        'Endocrinology & Metabolism',
        'Hematology',
        'Pulmonology & Respiratory Medicine',
        'Gastroenterology & Hepatology',
      ],
    },
    {
      number: 4,
      name: 'Primary Care & Mental Health',
      icon: Activity,
      description: 'Holistic care focusing on mental wellness, family health, and aging populations.',
      specialties: [
        'Psychiatry',
        'Psychology & Behavioral Health',
        'Nursing',
        'Family Medicine & Primary Care',
        'Geriatrics',
      ],
    },
  ];

  // Show chat interface when there are messages OR a pending question
  const showChat = messages.length > 0 || pendingQuestion !== null;

  return (
    <div className={`flex-1 flex flex-col max-w-4xl mx-auto w-full px-6 overflow-y-auto scrollbar-hide transition-all duration-500 min-h-0 ${
      !showChat 
        ? 'items-center justify-center pb-32' 
        : 'pb-32 pt-4'
    }`}>
      {!showChat ? (
        // Initial view - show welcome and clusters (even when one is selected)
        <>
          <div className="w-full mb-12 mt-12 animate-in slide-in-from-bottom-4 duration-700 text-center">
            <h1 className="text-6xl font-medium mb-2 tracking-tight">
              <span className="bg-gradient-to-r from-[#4285f4] via-[#9b72cb] to-[#d96570] bg-clip-text text-transparent">
                Welcome to MediRAG
              </span>
            </h1>
            <h4 className={`text-3xl font-medium tracking-tight ${
              theme === 'dark' ? 'text-[#444746]' : 'text-gray-400'
            }`}>
              Find your focus. Ask your questions.
            </h4>
          </div>

          <div className="grid grid-cols-2 md:grid-cols-4 gap-4 w-full animate-in fade-in zoom-in-95 duration-1000 delay-200">
            {clusters.map((cluster) => (
              <ClusterCard
                key={cluster.number}
                clusterNumber={cluster.number}
                name={cluster.name}
                icon={cluster.icon}
                description={cluster.description}
                onSelect={() => onClusterSelect(cluster.number)}
                isSelected={selectedCluster === cluster.number}
              />
            ))}
          </div>

          {/* Theme Toggle Button - Bottom Right */}
          <button
            onClick={toggleTheme}
            className={`fixed bottom-8 right-8 p-3 rounded-full shadow-lg transition-all z-10 ${
              theme === 'dark' 
                ? 'bg-[#1e1f20] border border-[#3c4043] hover:bg-[#282a2c] text-[#e3e3e3]' 
                : 'bg-white border border-gray-200 hover:bg-gray-100 text-gray-800'
            }`}
            title={theme === 'dark' ? 'Switch to light mode' : 'Switch to dark mode'}
          >
            {theme === 'dark' ? (
              <Moon className="w-5 h-5" />
            ) : (
              <Sun className="w-5 h-5" />
            )}
          </button>
        </>
      ) : (
        // Chat view - show messages at top, conversation flows down
        <div className="w-full space-y-6 animate-in fade-in slide-in-from-top-4 duration-500">
          {/* Render existing messages */}
          {messages.map((message, index) => (
            <div key={index} className="space-y-4">
              {/* User Question - Right aligned */}
              <div className="flex flex-col items-end">
                <div className={`max-w-[80%] rounded-2xl px-4 py-3 animate-in fade-in slide-in-from-right-4 duration-300 ${
                  theme === 'dark'
                    ? 'bg-[#1e1f20] border border-[#3c4043]'
                    : 'bg-gray-100 border border-gray-200'
                }`}>
                  <p className={`text-[15.5px] font-medium tracking-tight leading-relaxed ${
                    theme === 'dark' ? 'text-[#e3e3e3]' : 'text-gray-900'
                  }`}>{message.question}</p>
                </div>
                <button
                  onClick={() => handleCopy(message.question, 'question', index)}
                  className={`mt-1 p-1.5 rounded-lg transition-colors ${
                    theme === 'dark'
                      ? 'hover:bg-[#282a2c] text-[#8e9196] hover:text-[#e3e3e3]'
                      : 'hover:bg-gray-200 text-gray-500 hover:text-gray-700'
                  }`}
                  title="Copy question"
                >
                  {copiedIndex?.type === 'question' && copiedIndex?.index === index ? (
                    <Check className="w-4 h-4 text-green-400" />
                  ) : (
                    <Copy className="w-4 h-4" />
                  )}
                </button>
              </div>

              {/* Bot Answer - Left aligned */}
              <div className="flex flex-col items-start">
                <div className={`max-w-[80%] rounded-2xl px-4 py-3 animate-in fade-in slide-in-from-left-4 duration-300 ${
                  theme === 'dark'
                    ? 'bg-[#282a2c] border border-[#3c4043]'
                    : 'bg-gray-50 border border-gray-200'
                }`}>
                  <p className={`text-[15.5px] font-medium tracking-tight leading-relaxed whitespace-pre-wrap ${
                    theme === 'dark' ? 'text-[#e3e3e3]' : 'text-gray-900'
                  }`}>{message.answer}</p>
                </div>
                <button
                  onClick={() => handleCopy(message.answer, 'answer', index)}
                  className={`mt-1 p-1.5 rounded-lg transition-colors ${
                    theme === 'dark'
                      ? 'hover:bg-[#282a2c] text-[#8e9196] hover:text-[#e3e3e3]'
                      : 'hover:bg-gray-200 text-gray-500 hover:text-gray-700'
                  }`}
                  title="Copy answer"
                >
                  {copiedIndex?.type === 'answer' && copiedIndex?.index === index ? (
                    <Check className="w-4 h-4 text-green-400" />
                  ) : (
                    <Copy className="w-4 h-4" />
                  )}
                </button>
              </div>
            </div>
          ))}
          
          {/* Show pending question immediately if it exists */}
          {pendingQuestion && (
            <div className="space-y-4">
              {/* User Question - Right aligned */}
              <div className="flex flex-col items-end">
                <div className={`max-w-[80%] rounded-2xl px-4 py-3 animate-in fade-in slide-in-from-right-4 duration-300 ${
                  theme === 'dark'
                    ? 'bg-[#1e1f20] border border-[#3c4043]'
                    : 'bg-gray-100 border border-gray-200'
                }`}>
                  <p className={`text-[15.5px] font-medium tracking-tight leading-relaxed ${
                    theme === 'dark' ? 'text-[#e3e3e3]' : 'text-gray-900'
                  }`}>{pendingQuestion}</p>
                </div>
                <button
                  onClick={() => handleCopy(pendingQuestion, 'question', -1)}
                  className={`mt-1 p-1.5 rounded-lg transition-colors ${
                    theme === 'dark'
                      ? 'hover:bg-[#282a2c] text-[#8e9196] hover:text-[#e3e3e3]'
                      : 'hover:bg-gray-200 text-gray-500 hover:text-gray-700'
                  }`}
                  title="Copy question"
                >
                  {copiedIndex?.type === 'question' && copiedIndex?.index === -1 ? (
                    <Check className="w-4 h-4 text-green-400" />
                  ) : (
                    <Copy className="w-4 h-4" />
                  )}
                </button>
              </div>

              {/* Loading Answer - Left aligned */}
              <div className="flex justify-start">
                <div className={`max-w-[80%] rounded-2xl px-4 py-3 animate-in fade-in slide-in-from-left-4 duration-300 ${
                  theme === 'dark'
                    ? 'bg-[#282a2c] border border-[#3c4043]'
                    : 'bg-gray-50 border border-gray-200'
                }`}>
                  <div className={`flex items-center gap-2 ${
                    theme === 'dark' ? 'text-[#8e9196]' : 'text-gray-500'
                  }`}>
                    <Sparkles className="w-4 h-4 text-blue-400 animate-pulse" />
                    <span className="text-[14.5px] font-medium">{loadingMessages[currentLoadingMessage]}</span>
                  </div>
                </div>
              </div>
            </div>
          )}
        </div>
      )}
    </div>
  );
};
