/**
 * Persistence shape for MediRAG chat history (MongoDB `conversations` collection).
 */

import type { ObjectId } from 'mongodb';

export interface ConversationMessage {
  question: string;
  answer: string;
  timestamp: Date;
  verificationLevel?: number;
}

export interface ConversationDoc {
  _id?: ObjectId;
  userId: string;
  title: string;
  cluster: number | null;
  messages: ConversationMessage[];
  createdAt: Date;
  updatedAt: Date;
}

/** Fields returned for sidebar listings (minimal payload). */
export interface ConversationListItem {
  id: string;
  title: string;
  cluster: number | null;
  updatedAt: string;
  messageCount: number;
}
