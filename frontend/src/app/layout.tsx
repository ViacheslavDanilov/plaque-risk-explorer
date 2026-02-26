import type { Metadata } from "next";
import { DM_Serif_Display, IBM_Plex_Mono, Rajdhani } from "next/font/google";
import "./globals.css";

const rajdhani = Rajdhani({
  variable: "--font-rajdhani",
  weight: "700",
  subsets: ["latin"],
});

const dmSerifDisplay = DM_Serif_Display({
  variable: "--font-display",
  style: "normal",
  weight: "400",
  subsets: ["latin"],
});

const ibmPlexMono = IBM_Plex_Mono({
  variable: "--font-mono",
  weight: ["300", "400", "500", "600", "700"],
  subsets: ["latin"],
});

export const metadata: Metadata = {
  title: "Plaque Risk Explorer",
  description: "Clinical decision support for cardiovascular risk stratification with explainable ML and AI.",
  openGraph: {
    title: "Plaque Risk Explorer",
    description: "Clinical decision support for cardiovascular risk stratification with explainable ML and AI.",
  },
  twitter: {
    card: "summary_large_image",
    title: "Plaque Risk Explorer",
    description: "Clinical decision support for cardiovascular risk stratification with explainable ML and AI.",
  },
};

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html lang="en">
      <body
        className={`${rajdhani.variable} ${dmSerifDisplay.variable} ${ibmPlexMono.variable}`}
        suppressHydrationWarning
      >
        {children}
      </body>
    </html>
  );
}
