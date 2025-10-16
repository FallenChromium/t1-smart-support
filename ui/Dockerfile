# syntax=docker/dockerfile:1.7
# Multi-stage build for production-ready Next.js app with cached builds
FROM node:20-alpine AS base

# Install dependencies only when needed
FROM base AS deps
ENV NPM_CONFIG_CACHE=/root/.npm
RUN apk add --no-cache libc6-compat
WORKDIR /app

# Copy package files
COPY package.json package-lock.json* ./
RUN --mount=type=cache,target=/root/.npm npm ci

# Rebuild the source code only when needed
FROM base AS builder
ENV NPM_CONFIG_CACHE=/root/.npm
WORKDIR /app
ARG NEXT_PUBLIC_API_URL="http://backend:8000"
ENV NEXT_PUBLIC_API_URL=$NEXT_PUBLIC_API_URL
COPY --from=deps /app/node_modules ./node_modules
COPY . .

# Set environment variables for build
ENV NEXT_TELEMETRY_DISABLED=1

# Build the application
RUN --mount=type=cache,target=/root/.npm \
    --mount=type=cache,target=/app/.next/cache \
    npm run build

# Production image, copy all the files and run next
FROM base AS runner
WORKDIR /app

ENV NODE_ENV=production
ENV NEXT_TELEMETRY_DISABLED=1
ARG NEXT_PUBLIC_API_URL="http://backend:8000"
ENV NEXT_PUBLIC_API_URL=$NEXT_PUBLIC_API_URL

RUN addgroup --system --gid 1001 nodejs
RUN adduser --system --uid 1001 nextjs

# Copy necessary files
COPY --from=builder /app/public ./public
COPY --from=builder --chown=nextjs:nodejs /app/.next/standalone ./
COPY --from=builder --chown=nextjs:nodejs /app/.next/static ./.next/static

USER nextjs

EXPOSE 3000

ENV PORT=3000
ENV HOSTNAME="0.0.0.0"

CMD ["node", "server.js"]
