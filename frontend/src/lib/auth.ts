import { NextAuthOptions } from 'next-auth';
import CredentialsProvider from 'next-auth/providers/credentials';
import GoogleProvider from 'next-auth/providers/google';
import bcrypt from 'bcryptjs';
import clientPromise from './mongodb';

export const authOptions: NextAuthOptions = {
  providers: [
    GoogleProvider({
      clientId: process.env.GOOGLE_CLIENT_ID as string,
      clientSecret: process.env.GOOGLE_CLIENT_SECRET as string,
    }),
    CredentialsProvider({
      name: 'Credentials',
      credentials: {
        email: { label: 'Email', type: 'email' },
        password: { label: 'Password', type: 'password' },
      },
      async authorize(credentials) {
        if (!credentials?.email || !credentials?.password) {
          throw new Error('Please enter an email and password');
        }

        try {
          const client = await clientPromise;
          const db = client.db('medirag');
          const usersCollection = db.collection('users');

          const user = await usersCollection.findOne({
            email: credentials.email,
          });

          if (!user) {
            throw new Error('No user found with this email');
          }

          if (!user.password) {
            throw new Error('Please sign in with Google');
          }

          const isPasswordValid = await bcrypt.compare(
            credentials.password,
            user.password
          );

          if (!isPasswordValid) {
            throw new Error('Invalid password');
          }

          return {
            id: user._id.toString(),
            email: user.email,
            name: user.name,
            image: user.image || null,
          };
        } catch (error) {
          console.error('Auth error:', error);
          throw error;
        }
      },
    }),
  ],
  callbacks: {
    async signIn({ user, account }) {
      if (account?.provider === 'google') {
        try {
          const client = await clientPromise;
          const db = client.db('medirag');
          const usersCollection = db.collection('users');

          const existingUser = await usersCollection.findOne({
            email: user.email,
          });

          if (!existingUser) {
            await usersCollection.insertOne({
              email: user.email,
              name: user.name,
              image: user.image,
              provider: 'google',
              createdAt: new Date(),
            });
          }

          return true;
        } catch (error) {
          console.error('Error during Google sign in:', error);
          return false;
        }
      }
      return true;
    },
    async jwt({ token, user }) {
      if (user) {
        token.id = user.id;
      }
      return token;
    },
    async session({ session, token }) {
      if (session.user) {
        session.user.id = token.id;
      }
      return session;
    },
  },
  pages: {
    signIn: '/login',
    error: '/login',
  },
  session: {
    strategy: 'jwt',
  },
  secret: process.env.NEXTAUTH_SECRET,
};
