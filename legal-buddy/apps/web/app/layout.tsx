import type { Metadata } from "next";
import { Geist, Noto_Sans_Bengali } from "next/font/google";
import "./globals.css";

const geistSans = Geist({
  subsets: ["latin"],
  variable: "--font-geist-sans",
  display: "swap",
});

// Bengali fallback for the font stack: Geist covers Latin only, and the OS
// fallback for Bengali often lacks OpenType shaping — conjuncts (ক্ষ, ত্ত)
// split apart, vowel signs (ি ে) land in the wrong place, and some marks
// render as tofu. Noto Sans Bengali carries the full shaping tables.
const notoBengali = Noto_Sans_Bengali({
  subsets: ["bengali"],
  variable: "--font-noto-bengali",
  display: "swap",
});

export const metadata: Metadata = {
  title: "LegalBuddy — legal answers, cited",
  description:
    "Ask questions about Bangladesh statutory law and get answers grounded in the acts, cited to the section.",
};

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html
      lang="en"
      className={`${geistSans.variable} ${notoBengali.variable} h-full antialiased`}
    >
      <body className="min-h-full flex flex-col">{children}</body>
    </html>
  );
}
