/** @type {import('next').NextConfig} */
const nextConfig = {
  reactStrictMode: true,
  output: 'export',  // Enable static HTML export for single-container deployment
  distDir: 'out',    // Output directory

  // Disable features not supported in static export
  images: {
    unoptimized: true,
  },

  // Trailing slash for consistent routing
  trailingSlash: true,

  // Environment variables injected at build time
  env: {
    NEXT_PUBLIC_API_BASE_URL: process.env.NEXT_PUBLIC_API_BASE_URL || '/api',
  },
};

export default nextConfig;
