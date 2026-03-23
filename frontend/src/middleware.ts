import { withAuth } from 'next-auth/middleware';
import { NextResponse } from 'next/server';

export default withAuth(
  function middleware(req) {
    return NextResponse.next();
  },
  {
    callbacks: {
      authorized: ({ token }) => !!token,
    },
    pages: {
      signIn: '/login',
    },
  }
);

// Only protect specific routes that require authentication
// Homepage (/) is publicly accessible for guest users
export const config = {
  matcher: [
    /*
     * Match only protected routes:
     * - /dashboard (if exists)
     * - /profile (if exists)
     * - /settings (if exists)
     * Add specific protected routes here as needed
     * 
     * Homepage and all other routes are accessible to guests
     */
    '/dashboard/:path*',
    '/profile/:path*',
    '/settings/:path*',
  ],
};
