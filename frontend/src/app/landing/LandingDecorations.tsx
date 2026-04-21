'use client';

import React from 'react';
import { useTheme } from '@/contexts/ThemeContext';

function Cross({ size, color }: { size: number; color: string }) {
  return (
    <svg width={size} height={size} viewBox="0 0 28 28" fill={color} aria-hidden>
      <rect x="11" y="0" width="6" height="28" rx="2" />
      <rect x="0" y="11" width="28" height="6" rx="2" />
    </svg>
  );
}

function Hexagon({ size, color }: { size: number; color: string }) {
  return (
    <svg width={size} height={size} viewBox="0 0 100 100" fill="none" aria-hidden>
      <polygon points="50,5 93,27.5 93,72.5 50,95 7,72.5 7,27.5" stroke={color} strokeWidth="5" />
    </svg>
  );
}

function DotGrid({ color }: { color: string }) {
  return (
    <svg width="72" height="72" viewBox="0 0 72 72" fill={color} aria-hidden>
      {[0, 1, 2, 3].flatMap((r) =>
        [0, 1, 2, 3].map((c) => (
          <circle key={`${r}-${c}`} cx={c * 22 + 3} cy={r * 22 + 3} r="2.5" />
        ))
      )}
    </svg>
  );
}

type ShapeConfig = {
  shape: 'cross' | 'hex' | 'dots';
  size?: number;
  top?: string;
  left?: string;
  right?: string;
  rotate: number;
  color: 'teal' | 'blue';
};

const shapes: ShapeConfig[] = [
  { shape: 'hex',   size: 72, top: '6%',  right: '4%',  rotate: 15,  color: 'teal' },
  { shape: 'cross', size: 22, top: '11%', left: '7%',   rotate: -12, color: 'teal' },
  { shape: 'dots',            top: '20%', right: '10%', rotate: 0,   color: 'blue' },
  { shape: 'hex',   size: 48, top: '33%', left: '2%',   rotate: 30,  color: 'blue' },
  { shape: 'cross', size: 34, top: '40%', right: '3%',  rotate: 20,  color: 'teal' },
  { shape: 'cross', size: 18, top: '53%', left: '42%',  rotate: -5,  color: 'teal' },
  { shape: 'dots',            top: '60%', left: '5%',   rotate: 15,  color: 'teal' },
  { shape: 'hex',   size: 60, top: '68%', right: '7%',  rotate: -10, color: 'blue' },
  { shape: 'cross', size: 26, top: '78%', left: '11%',  rotate: 8,   color: 'teal' },
  { shape: 'dots',            top: '86%', right: '6%',  rotate: -20, color: 'blue' },
  { shape: 'cross', size: 20, top: '25%', left: '28%',  rotate: 10,  color: 'blue' },
  { shape: 'hex',   size: 44, top: '47%', right: '20%', rotate: -8,  color: 'teal' },
  { shape: 'dots',            top: '72%', left: '30%',  rotate: 5,   color: 'blue' },
  { shape: 'cross', size: 16, top: '92%', left: '55%',  rotate: -14, color: 'teal' },
];

export function LandingDecorations() {
  const { theme } = useTheme();
  const isDark = theme === 'dark';

  const orange = isDark ? 'rgba(249,115,22,0.32)' : 'rgba(234,88,12,0.16)';
  const amber = isDark ? 'rgba(251,146,60,0.26)' : 'rgba(249,115,22,0.13)';
  const c = (color: 'teal' | 'blue') => (color === 'teal' ? orange : amber);

  return (
    <div aria-hidden className="pointer-events-none absolute inset-0 overflow-hidden" style={{ zIndex: 0 }}>
      {shapes.map((s, i) => (
        <div
          key={i}
          className="absolute"
          style={{
            top: s.top,
            left: s.left,
            right: s.right,
            transform: `rotate(${s.rotate}deg)`,
          }}
        >
          {s.shape === 'cross' && <Cross size={s.size ?? 28} color={c(s.color)} />}
          {s.shape === 'hex'   && <Hexagon size={s.size ?? 48} color={c(s.color)} />}
          {s.shape === 'dots'  && <DotGrid color={c(s.color)} />}
        </div>
      ))}
    </div>
  );
}
