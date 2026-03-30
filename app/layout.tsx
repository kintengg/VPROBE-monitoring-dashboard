import type { Metadata } from 'next'
import { Inter } from 'next/font/google'
import { Analytics } from '@vercel/analytics/next'
import './globals.css'
import { Sidebar } from '@/components/sidebar'
import { LoadingProvider } from '@/components/ui/walking-loader'
import { UploadQueueProvider } from '@/components/uploads/upload-queue-provider'

const inter = Inter({ 
  subsets: ["latin"],
  variable: '--font-inter'
});

export const metadata: Metadata = {
  title: 'ALIVE Engine - Advanced Pedestrian Surveillance VMS',
  description: 'Advanced Pedestrian Surveillance Video Management System powered by ByteTrack',
  generator: 'v0.app',
}

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode
}>) {
  return (
    <html lang="en" className="dark">
      <body className={`${inter.variable} font-sans antialiased bg-[#1C1C1E] text-[#F5F5F7]`}>
        <LoadingProvider>
          <UploadQueueProvider>
            <div className="flex h-screen overflow-hidden">
              <Sidebar />
              <main className="flex-1 overflow-auto">
                {children}
              </main>
            </div>
          </UploadQueueProvider>
        </LoadingProvider>
        <Analytics />
      </body>
    </html>
  )
}
