'use client';

import React, { useState } from 'react';
import { Sidebar } from '@/components/Sidebar/Sidebar';
import { Header } from '@/components/Header/Header';
import { ChatArea } from '@/components/ChatArea/ChatArea';
import { InputArea } from '@/components/InputArea/InputArea';
import { useTheme } from '@/contexts/ThemeContext';

interface Message {
  question: string;
  answer: string;
  timestamp: Date;
}

type SelectedCluster = 1 | 2 | 3 | 4;

type QueryRequestPayload =
  | { query: string; k?: number; alpha?: number }
  | { query: string; top_k: number }
  | { question: string };

export default function Home() {
  const [isSidebarOpen, setIsSidebarOpen] = useState(true);
  const [selectedCluster, setSelectedCluster] = useState<number | null>(null);
  const [messages, setMessages] = useState<Message[]>([]);
  const [isLoading, setIsLoading] = useState(false);
  const [pendingQuestion, setPendingQuestion] = useState<string | null>(null);
  const { theme } = useTheme();

  // Keep backend routing in one place to avoid accidental mismatches.
  // NOTE: Cluster 4 must use the primary care backend on port 8003.
  const BACKEND_URL_BY_CLUSTER: Record<number, string> = {
    1: 'http://127.0.0.1:8000/query',
    2: 'http://127.0.0.1:8001/query2',
    3: 'http://127.0.0.1:8002/rag/answer-verified',
    4: 'http://127.0.0.1:8003/query4',
  };

  // Build request bodies per cluster. By default, keep the current payload shape
  // so existing backends keep working; adjust individual clusters as needed.
  const REQUEST_BODY_BY_CLUSTER: Record<SelectedCluster, (question: string) => QueryRequestPayload> = {
    1: (q: string) => ({ question: q }),
    2: (q: string) => ({ query: q, k: 5, alpha: 0.5 }),
    // Cluster 3 will be wired later; keep a backward-compatible default for now.
    3: (q: string) => ({ query: q, k: 5, gen_max_length: 256, temperature: 0.0 }),
    4: (q: string) => ({ query: q, top_k: 5 }),
  };

  const handleClusterSelect = (clusterNumber: number) => {
    setSelectedCluster(selectedCluster === clusterNumber ? null : clusterNumber);
    // Clear messages and pending question when switching clusters
    if (selectedCluster !== clusterNumber) {
      setMessages([]);
      setPendingQuestion(null);
    }
  };

  const handleNewChat = () => {
    setSelectedCluster(null);
    setMessages([]);
    setPendingQuestion(null);
  };

  const getSelectedClusterName = () => {
    const clusters = [
      { number: 1, name: 'Neurosciences' },
      { number: 2, name: 'Cardiovascular' },
      { number: 3, name: 'Internal Medicine' },
      { number: 4, name: 'Primary Care & Mental Health' },
    ];
    const cluster = clusters.find(c => c.number === selectedCluster);
    return cluster?.name || null;
  };

  const handleSendMessage = async (question: string) => {
    if (!question.trim() || !selectedCluster) return;

    // Immediately show the question and transition to conversation mode
    setPendingQuestion(question);
    setIsLoading(true);
    
      try {
      // Map clusters to backends.
      const backendUrl = BACKEND_URL_BY_CLUSTER[selectedCluster] ?? null;

      if (!backendUrl) {
        throw new Error('Backend not available for this cluster');
      }

      const requestBodyBuilder = REQUEST_BODY_BY_CLUSTER[selectedCluster as SelectedCluster];
      const requestBody = requestBodyBuilder
        ? requestBodyBuilder(question)
        : ({ query: question, k: 5, alpha: 0.5 } satisfies QueryRequestPayload);

      // Backend expects { query, k, alpha }
      const response = await fetch(backendUrl, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify(requestBody),
      });

      if (!response.ok) {
        throw new Error('Failed to get response from backend');
      }

      const data = await response.json();

      // Add the message with answer and clear pending question
      setMessages(prev => [
        ...prev,
        {
          question,
          answer: data.answer,
          timestamp: new Date(),
        },
      ]);
      setPendingQuestion(null);
    } catch (error) {
      console.error('Error sending message:', error);
      // Add error message and clear pending question
      setMessages(prev => [
        ...prev,
        {
          question,
          answer: 'Sorry, I encountered an error. Please try again.',
          timestamp: new Date(),
        },
      ]);
      setPendingQuestion(null);
    } finally {
      setIsLoading(false);
    }
  };

  return (
    <div className={`flex h-screen font-sans overflow-hidden ${
      theme === 'dark' 
        ? 'bg-[#131314] text-[#e3e3e3]' 
        : 'bg-white text-gray-900'
    }`}>
      <Sidebar 
        isOpen={isSidebarOpen} 
        onToggle={() => setIsSidebarOpen(!isSidebarOpen)}
        onNewChat={handleNewChat}
      />
      <main className="flex-1 flex flex-col relative overflow-hidden">
        <Header />
        <ChatArea 
          selectedCluster={selectedCluster}
          messages={messages}
          onClusterSelect={handleClusterSelect}
          pendingQuestion={pendingQuestion}
          isLoading={isLoading}
        />
        <InputArea 
          onSend={handleSendMessage}
          isLoading={isLoading}
          disabled={!selectedCluster}
          selectedClusterName={getSelectedClusterName()}
          hasMessages={messages.length > 0 || pendingQuestion !== null}
        />
      </main>
    </div>
  );
}
