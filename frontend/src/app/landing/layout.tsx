import type { Metadata } from 'next';

export const metadata: Metadata = {
  title: 'MediRAG — Clinical knowledge, grounded in evidence',
  description:
    'Four retrieval-augmented pipelines for neurosciences, cardiovascular care, internal medicine, and primary care & mental health.',
};

export default function LandingLayout({
  children,
}: Readonly<{ children: React.ReactNode }>) {
  return children;
}
