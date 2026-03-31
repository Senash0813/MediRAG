import type { Collection } from 'mongodb';
import clientPromise from '@/lib/mongodb';
import type { ConversationDoc } from '@/types/conversation';

export const MONGODB_DB_NAME = 'medirag';
export const CONVERSATIONS_COLLECTION = 'conversations';

let indexesEnsured = false;

export async function getConversationsCollection(): Promise<
  Collection<ConversationDoc>
> {
  const client = await clientPromise;
  return client
    .db(MONGODB_DB_NAME)
    .collection<ConversationDoc>(CONVERSATIONS_COLLECTION);
}

/**
 * Creates indexes if missing. Safe to call from API handlers; runs at most once per process after first successful call.
 */
export async function ensureConversationIndexes(): Promise<void> {
  if (indexesEnsured) return;
  const col = await getConversationsCollection();
  await col.createIndex({ userId: 1, updatedAt: -1 });
  indexesEnsured = true;
}
