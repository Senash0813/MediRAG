import { NextResponse } from 'next/server';
import { getServerSession } from 'next-auth';
import { ObjectId } from 'mongodb';
import { authOptions } from '@/lib/auth';
import {
  ensureConversationIndexes,
  getConversationsCollection,
} from '@/lib/conversations';

type RouteContext = { params: Promise<{ id: string }> };

function serializeConversation(doc: {
  _id: ObjectId;
  title: string;
  cluster: number | null;
  messages: Array<{
    question: string;
    answer: string;
    timestamp: Date;
    verificationLevel?: number;
  }>;
  createdAt: Date;
  updatedAt: Date;
}) {
  return {
    id: doc._id.toString(),
    title: doc.title,
    cluster: doc.cluster,
    messages: doc.messages.map((m) => ({
      question: m.question,
      answer: m.answer,
      timestamp:
        m.timestamp instanceof Date
          ? m.timestamp.toISOString()
          : new Date(m.timestamp as string).toISOString(),
      verificationLevel: m.verificationLevel,
    })),
    createdAt:
      doc.createdAt instanceof Date
        ? doc.createdAt.toISOString()
        : new Date(doc.createdAt as string).toISOString(),
    updatedAt:
      doc.updatedAt instanceof Date
        ? doc.updatedAt.toISOString()
        : new Date(doc.updatedAt as string).toISOString(),
  };
}

export async function GET(_request: Request, context: RouteContext) {
  try {
    const session = await getServerSession(authOptions);
    const userId = session?.user?.id;
    if (!userId) {
      return NextResponse.json({ error: 'Unauthorized' }, { status: 401 });
    }

    const { id } = await context.params;
    if (!ObjectId.isValid(id)) {
      return NextResponse.json({ error: 'Not found' }, { status: 404 });
    }

    await ensureConversationIndexes();
    const col = await getConversationsCollection();
    const doc = await col.findOne({
      _id: new ObjectId(id),
      userId,
    });

    if (!doc || !doc._id) {
      return NextResponse.json({ error: 'Not found' }, { status: 404 });
    }

    const messages = Array.isArray(doc.messages) ? doc.messages : [];
    return NextResponse.json(serializeConversation({ ...doc, messages }));
  } catch (e) {
    console.error('GET /api/conversations/[id]:', e);
    return NextResponse.json(
      { error: 'Failed to load conversation' },
      { status: 500 }
    );
  }
}
