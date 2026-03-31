import { NextResponse } from 'next/server';
import { getServerSession } from 'next-auth';
import { authOptions } from '@/lib/auth';
import {
  ensureConversationIndexes,
  getConversationsCollection,
} from '@/lib/conversations';
import type { ConversationListItem } from '@/types/conversation';

export async function GET() {
  try {
    const session = await getServerSession(authOptions);
    const userId = session?.user?.id;
    if (!userId) {
      return NextResponse.json({ error: 'Unauthorized' }, { status: 401 });
    }

    await ensureConversationIndexes();
    const col = await getConversationsCollection();

    const docs = await col
      .find({ userId })
      .sort({ updatedAt: -1 })
      .project({
        title: 1,
        cluster: 1,
        updatedAt: 1,
        messages: 1,
      })
      .toArray();

    const items: ConversationListItem[] = docs.map((doc) => ({
      id: doc._id!.toString(),
      title: doc.title || 'New chat',
      cluster: doc.cluster ?? null,
      updatedAt:
        doc.updatedAt instanceof Date
          ? doc.updatedAt.toISOString()
          : new Date(doc.updatedAt as string).toISOString(),
      messageCount: Array.isArray(doc.messages) ? doc.messages.length : 0,
    }));

    return NextResponse.json({ conversations: items });
  } catch (e) {
    console.error('GET /api/conversations:', e);
    return NextResponse.json(
      { error: 'Failed to list conversations' },
      { status: 500 }
    );
  }
}

export async function POST(request: Request) {
  try {
    const session = await getServerSession(authOptions);
    const userId = session?.user?.id;
    if (!userId) {
      return NextResponse.json({ error: 'Unauthorized' }, { status: 401 });
    }

    let body: { title?: unknown; cluster?: unknown } = {};
    try {
      body = await request.json();
    } catch {
      body = {};
    }

    let title = '';
    if (typeof body.title === 'string') {
      title = body.title.slice(0, 200).trim();
    }

    const ALLOWED_CLUSTERS = new Set([1, 2, 3, 4]);
    let cluster: number | null = null;
    if (
      typeof body.cluster === 'number' &&
      Number.isInteger(body.cluster) &&
      ALLOWED_CLUSTERS.has(body.cluster)
    ) {
      cluster = body.cluster;
    }

    await ensureConversationIndexes();
    const col = await getConversationsCollection();

    const now = new Date();
    const result = await col.insertOne({
      userId,
      title,
      cluster,
      messages: [],
      createdAt: now,
      updatedAt: now,
    });

    return NextResponse.json(
      { id: result.insertedId.toString(), title, cluster, messageCount: 0 },
      { status: 201 }
    );
  } catch (e) {
    console.error('POST /api/conversations:', e);
    return NextResponse.json(
      { error: 'Failed to create conversation' },
      { status: 500 }
    );
  }
}
