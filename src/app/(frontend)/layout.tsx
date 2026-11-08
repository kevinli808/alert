import React from 'react'
import type { Metadata } from 'next'
import './styles.css'

const title = 'Alert'
const description = 'Making environmental sounds visible through real-time visual subtitles.'
const siteURL =
  process.env.NEXT_PUBLIC_SITE_URL ||
  (process.env.VERCEL_URL ? `https://${process.env.VERCEL_URL}` : 'http://localhost:3000')

export const metadata: Metadata = {
  metadataBase: new URL(siteURL),
  title,
  description,
  icons: { icon: '/favicon.svg' },
  openGraph: { title, description },
}

export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <html lang="en">
      <body className="min-h-screen bg-[#09090b] text-zinc-100 antialiased">
        <a className="skip-link" href="#root">
          Skip to content
        </a>
        <div id="root" tabIndex={-1}>
          {children}
        </div>
      </body>
    </html>
  )
}
