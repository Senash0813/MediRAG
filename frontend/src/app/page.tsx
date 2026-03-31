'use client';

import React, { useState, useCallback, useEffect, useRef } from 'react';
import { useSession } from 'next-auth/react';
import { Sidebar } from '@/components/Sidebar/Sidebar';
import { Header } from '@/components/Header/Header';
import { ChatArea } from '@/components/ChatArea/ChatArea';
import { InputArea } from '@/components/InputArea/InputArea';
import { LoginModal, SignupModal } from '@/components/Auth';
import { useTheme } from '@/contexts/ThemeContext';

interface Message {
  question: string;
  answer: string;
  timestamp: Date;
  cluster: number;
  verificationLevel?: number;
}

type SelectedCluster = 1 | 2 | 3 | 4;

type QueryRequestPayload =
  | { query: string; k?: number; alpha?: number }
  | { query: string; top_k: number }
  | { question: string };

export default function Home() {
  const { data: session, status } = useSession();
  const [isSidebarOpen, setIsSidebarOpen] = useState(true);
  const [selectedCluster, setSelectedCluster] = useState<number | null>(null);
  const [messages, setMessages] = useState<Message[]>([]);
  const [isLoading, setIsLoading] = useState(false);
  const [pendingQuestion, setPendingQuestion] = useState<string | null>(null);
  const [isLoginModalOpen, setIsLoginModalOpen] = useState(false);
  const [isSignupModalOpen, setIsSignupModalOpen] = useState(false);
  const [activeConversationId, setActiveConversationId] = useState<string | null>(
    null
  );
  const activeConversationIdRef = useRef<string | null>(null);
  const [historyVersion, setHistoryVersion] = useState(0);
  const { theme } = useTheme();

  const setConversationId = useCallback((id: string | null) => {
    activeConversationIdRef.current = id;
    setActiveConversationId(id);
  }, []);

  useEffect(() => {
    if (status === 'unauthenticated') {
      setConversationId(null);
      setMessages([]);
      setPendingQuestion(null);
      setSelectedCluster(null);
      setIsLoading(false);
    }
  }, [status, setConversationId]);

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
    3: (q: string) => ({ query: q, k: 5, gen_max_length: 256, temperature: 0 }),
    4: (q: string) => ({ query: q, top_k: 5 }),
  };

  const handleClusterSelect = (clusterNumber: number) => {
    setSelectedCluster(selectedCluster === clusterNumber ? null : clusterNumber);
    // Clear messages and pending question when switching clusters
      if (selectedCluster !== clusterNumber) {
      setMessages([]);
      setPendingQuestion(null);
      setConversationId(null);
    }
  };

  const handleNewChat = () => {
    setSelectedCluster(null);
    setMessages([]);
    setPendingQuestion(null);
    setConversationId(null);
  };

  const loadConversation = useCallback(
    async (id: string) => {
      if (status !== 'authenticated') return;
      const r = await fetch(`/api/conversations/${id}`);
      if (!r.ok) return;
      const data = (await r.json()) as {
        id: string;
        cluster: number | null;
        messages: Array<{
          question: string;
          answer: string;
          timestamp: string;
          verificationLevel?: number;
        }>;
      };
      const cluster =
        typeof data.cluster === 'number' && Number.isFinite(data.cluster)
          ? data.cluster
          : null;
      setConversationId(data.id);
      setSelectedCluster(cluster);
      setMessages(
        (data.messages || []).map((m) => ({
          question: m.question,
          answer: m.answer,
          timestamp: new Date(m.timestamp),
          cluster: cluster ?? 0,
          verificationLevel: m.verificationLevel,
        }))
      );
      setPendingQuestion(null);
    },
    [status, setConversationId]
  );

  const persistExchange = async (
    question: string,
    answer: string,
    cluster: number,
    verificationLevel?: number
  ) => {
    if (status !== 'authenticated' || !session?.user?.id) return;
    try {
      let cid = activeConversationIdRef.current;
      if (!cid) {
        const res = await fetch('/api/conversations', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ cluster }),
        });
        if (!res.ok) throw new Error('Failed to create conversation');
        const created = (await res.json()) as { id: string };
        cid = created.id;
        setConversationId(cid);
      }
      const res = await fetch(`/api/conversations/${cid}/messages`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          question,
          answer,
          ...(verificationLevel !== undefined ? { verificationLevel } : {}),
        }),
      });
      if (!res.ok) throw new Error('Failed to save message');
      setHistoryVersion((v) => v + 1);
    } catch (e) {
      console.error('Failed to persist conversation:', e);
    }
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

      // Format the answer based on cluster-specific response structure
      let formattedAnswer = '';
      let verificationLevel: number | undefined;
      
      if (selectedCluster === 4) {
        // Cluster 4 (Primary Care) returns: direct_answer, evidence_summary, limitations
        formattedAnswer = data.direct_answer ?? '';
        if (data.evidence_summary) {
          formattedAnswer += '\n\n**Evidence Summary:**\n' + data.evidence_summary;
        }
        if (data.limitations) {
          formattedAnswer += '\n\n**Limitations:**\n' + data.limitations;
        }

		// Capture verification level (1-4) for UI warnings, if provided
		if (typeof data.verification_level === 'number') {
			verificationLevel = data.verification_level;
		}
      } else {
        // Other clusters return a simple 'answer' field
        formattedAnswer = data.answer || data.direct_answer || 'No answer received';
      }

      // Add the message with answer and clear pending question
      setMessages(prev => [
        ...prev,
        {
          question,
          answer: formattedAnswer,
          timestamp: new Date(),
          cluster: selectedCluster,
          verificationLevel,
        },
      ]);
      setPendingQuestion(null);
      void persistExchange(
        question,
        formattedAnswer,
        selectedCluster,
        verificationLevel
      );
    } catch (error) {
      console.error('Error sending message:', error);
      // Add error message and clear pending question
      setMessages(prev => [
        ...prev,
        {
          question,
          answer: 'Sorry, I encountered an error. Please try again.',
          cluster: selectedCluster || 0,
          timestamp: new Date(),
        },
      ]);
      setPendingQuestion(null);
    } finally {
      setIsLoading(false);
    }
  };

  return (
    <div className={`flex h-screen min-h-0 font-sans overflow-hidden ${
      theme === 'dark' 
        ? 'bg-[#131314] text-[#e3e3e3]' 
        : 'bg-background text-gray-900'
    }`}>
      <Sidebar
        isOpen={isSidebarOpen}
        onToggle={() => setIsSidebarOpen(!isSidebarOpen)}
        onNewChat={handleNewChat}
        onLoginClick={() => setIsLoginModalOpen(true)}
        historyVersion={historyVersion}
        activeConversationId={activeConversationId}
        onOpenConversation={loadConversation}
      />
      <main className="flex-1 flex flex-col relative overflow-hidden">
        <Header onLoginClick={() => setIsLoginModalOpen(true)} />
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

      {/* Login Modal */}
      <LoginModal 
        isOpen={isLoginModalOpen}
        onClose={() => setIsLoginModalOpen(false)}
        onSwitchToSignup={() => {
          setIsLoginModalOpen(false);
          setIsSignupModalOpen(true);
        }}
        onSuccess={() => {
          console.log('Login successful');
        }}
      />

      {/* Signup Modal */}
      <SignupModal 
        isOpen={isSignupModalOpen}
        onClose={() => setIsSignupModalOpen(false)}
        onSwitchToLogin={() => {
          setIsSignupModalOpen(false);
          setIsLoginModalOpen(true);
        }}
        onSuccess={() => {
          console.log('Signup successful');
        }}
      />
    </div>
  );
}
