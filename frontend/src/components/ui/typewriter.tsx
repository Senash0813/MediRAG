'use client';

import { useEffect, useRef, useState } from 'react';

export interface TypewriterSegment {
  text: string;
  className?: string;
}

export interface TypewriterSequenceProps {
  segments: TypewriterSegment[];
  speed?: number;
  delayBetweenSegments?: number;
  cursor?: boolean;
  cursorChar?: string;
  onComplete?: () => void;
}

/**
 * Types each segment in order; completed text stays visible (no delete / no loop).
 */
export function TypewriterSequence({
  segments,
  speed = 80,
  delayBetweenSegments = 400,
  cursor = true,
  cursorChar = '|',
  onComplete,
}: TypewriterSequenceProps) {
  const [segmentIndex, setSegmentIndex] = useState(0);
  const [charIndex, setCharIndex] = useState(0);
  const [done, setDone] = useState(false);
  const [showCursor, setShowCursor] = useState(true);
  const finishedRef = useRef(false);
  const onCompleteRef = useRef(onComplete);

  useEffect(() => {
    onCompleteRef.current = onComplete;
  }, [onComplete]);

  const currentText = segments[segmentIndex]?.text ?? '';

  useEffect(() => {
    if (done || segments.length === 0) return;

    let cancelled = false;
    let timeoutId: number | undefined;

    if (charIndex < currentText.length) {
      timeoutId = window.setTimeout(() => {
        if (!cancelled) setCharIndex((c) => c + 1);
      }, speed);
    } else if (segmentIndex < segments.length - 1) {
      timeoutId = window.setTimeout(() => {
        if (cancelled) return;
        setSegmentIndex((s) => s + 1);
        setCharIndex(0);
      }, delayBetweenSegments);
    } else if (!finishedRef.current) {
      finishedRef.current = true;
      timeoutId = window.setTimeout(() => {
        if (cancelled) return;
        setDone(true);
        onCompleteRef.current?.();
      }, 0);
    }

    return () => {
      cancelled = true;
      if (timeoutId !== undefined) window.clearTimeout(timeoutId);
    };
  }, [
    charIndex,
    currentText.length,
    delayBetweenSegments,
    done,
    segmentIndex,
    segments,
    speed,
  ]);

  useEffect(() => {
    if (!cursor || done) return;
    const id = window.setInterval(() => {
      setShowCursor((p) => !p);
    }, 500);
    return () => window.clearInterval(id);
  }, [cursor, done]);

  return (
    <span className="inline-flex w-full max-w-full flex-col items-start">
      {segments.map((seg, i) => {
        if (i > segmentIndex) return null;
        const full = seg.text;
        const visible = i < segmentIndex ? full : full.slice(0, charIndex);
        const active = i === segmentIndex && !done;
        return (
          <span key={i} className={`inline-flex min-h-[1.2em] items-baseline ${seg.className ?? ''}`}>
            {visible}
            {cursor && active && (
              <span
                className="ml-0.5 inline transition-opacity duration-75"
                style={{ opacity: showCursor ? 1 : 0 }}
                aria-hidden
              >
                {cursorChar}
              </span>
            )}
          </span>
        );
      })}
    </span>
  );
}

/** Original cycling typewriter (types, deletes, next word). */
export interface TypewriterProps {
  words: string[];
  speed?: number;
  delayBetweenWords?: number;
  cursor?: boolean;
  cursorChar?: string;
}

export function Typewriter({
  words,
  speed = 100,
  delayBetweenWords = 2000,
  cursor = true,
  cursorChar = '|',
}: TypewriterProps) {
  const [displayText, setDisplayText] = useState('');
  const [isDeleting, setIsDeleting] = useState(false);
  const [wordIndex, setWordIndex] = useState(0);
  const [charIndex, setCharIndex] = useState(0);
  const [showCursor, setShowCursor] = useState(true);

  const currentWord = words[wordIndex] ?? '';

  useEffect(() => {
    if (words.length === 0) return;

    const delay = isDeleting ? speed / 2 : speed;
    let timeoutId: ReturnType<typeof setTimeout>;

    if (!isDeleting) {
      if (charIndex < currentWord.length) {
        timeoutId = setTimeout(() => {
          setDisplayText(currentWord.slice(0, charIndex + 1));
          setCharIndex((c) => c + 1);
        }, delay);
      } else {
        timeoutId = setTimeout(() => setIsDeleting(true), delayBetweenWords);
      }
    } else if (charIndex > 0) {
      timeoutId = setTimeout(() => {
        setDisplayText(currentWord.slice(0, charIndex - 1));
        setCharIndex((c) => c - 1);
      }, delay);
    } else {
      timeoutId = setTimeout(() => {
        setIsDeleting(false);
        setWordIndex((prev) => (prev + 1) % words.length);
      }, speed);
    }

    return () => clearTimeout(timeoutId);
  }, [
    charIndex,
    currentWord,
    delayBetweenWords,
    isDeleting,
    speed,
    wordIndex,
    words.length,
  ]);

  useEffect(() => {
    if (!cursor) return;
    const id = setInterval(() => setShowCursor((p) => !p), 500);
    return () => clearInterval(id);
  }, [cursor]);

  return (
    <span className="inline-block">
      <span>
        {displayText}
        {cursor && (
          <span
            className="ml-1 transition-opacity duration-75"
            style={{ opacity: showCursor ? 1 : 0 }}
          >
            {cursorChar}
          </span>
        )}
      </span>
    </span>
  );
}
