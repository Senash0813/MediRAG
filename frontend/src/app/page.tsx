'use client';

import React, { useState } from 'react';
import { Sidebar } from '@/components/Sidebar/Sidebar';
import { Header } from '@/components/Header/Header';
import { ChatArea } from '@/components/ChatArea/ChatArea';
import { InputArea } from '@/components/InputArea/InputArea';

interface Message {
  question: string;
  answer: string;
  timestamp: Date;
}

export default function Home() {
  const [isSidebarOpen, setIsSidebarOpen] = useState(true);
  const [selectedCluster, setSelectedCluster] = useState<number | null>(null);
  const [messages, setMessages] = useState<Message[]>([]);
  const [isLoading, setIsLoading] = useState(false);
  const [pendingQuestion, setPendingQuestion] = useState<string | null>(null);

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
      // Map Neurosciences (cluster 1) and Cardiology (cluster 2) to backends
      // Update these URLs if your services run on different ports/hosts
      const backendUrl = selectedCluster === 1
        ? 'http://127.0.0.1:8000/query'
        : selectedCluster === 2
          ? 'http://127.0.0.1:8000/query2'
          : null;

      if (!backendUrl) {
        throw new Error('Backend not available for this cluster');
      }

      // Backend expects { query, k, alpha }
      const response = await fetch(backendUrl, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({ query: question, k: 5, alpha: 0.5 }),
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
    <div className="flex h-screen bg-[#131314] text-[#e3e3e3] font-sans overflow-hidden">
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
