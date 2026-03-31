import { NextResponse } from 'next/server';
import { getServerSession } from 'next-auth';
import { ObjectId } from 'mongodb';
import { authOptions } from '@/lib/auth';
import {
  ensureConversationIndexes,
  getConversationsCollection,
} from '@/lib/conversations';

type RouteContext = { params: Promise<{ id: string }> };

const MAX_FIELD_LENGTH = 50_000;

function asTrimmedString(value: unknown, max: number): string | null {
  if (typeof value !== 'string') return null;
  const t = value.trim();
  if (!t) return null;
  return t.slice(0, max);
}

export async function POST(request: Request, context: RouteContext) {
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

    const body = await request.json().catch(() => null);
    if (!body || typeof body !== 'object') {
      return NextResponse.json({ error: 'Invalid body' }, { status: 400 });
    }

    const question = asTrimmedString(
      (body as { question?: unknown }).question,
      MAX_FIELD_LENGTH
    );
    const answer = asTrimmedString(
      (body as { answer?: unknown }).answer,
      MAX_FIELD_LENGTH
    );
    if (!question || !answer) {
      return NextResponse.json(
        { error: 'question and answer are required' },
        { status: 400 }
      );
    }

    let verificationLevel: number | undefined;
    const vl = (body as { verificationLevel?: unknown }).verificationLevel;
    if (
      typeof vl === 'number' &&
      Number.isInteger(vl) &&
      vl >= 1 &&
      vl <= 4
    ) {
      verificationLevel = vl;
    }

    await ensureConversationIndexes();
    const col = await getConversationsCollection();
    const oid = new ObjectId(id);

    const doc = await col.findOne({ _id: oid, userId });
    if (!doc) {
      return NextResponse.json({ error: 'Not found' }, { status: 404 });
    }

    const title =
      doc.title?.trim() ||
      (question.length > 80 ? `${question.slice(0, 77)}...` : question);

    const newMessage = {
      question,
      answer,
      timestamp: new Date(),
      ...(verificationLevel !== undefined ? { verificationLevel } : {}),
    };

    await col.updateOne(
      { _id: oid, userId },
      {
        $push: { messages: newMessage },
        $set: { updatedAt: new Date(), title },
      }
    );

    return NextResponse.json({ ok: true, messageCount: doc.messages.length + 1 });
  } catch (e) {
    console.error('POST /api/conversations/[id]/messages:', e);
    return NextResponse.json(
      { error: 'Failed to append message' },
      { status: 500 }
    );
  }
}
