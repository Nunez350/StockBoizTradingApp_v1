import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

// Dev server proxies API and report requests to `quantlab api serve` (port 8000).
// In production the API serves the built bundle from web/dist itself.
export default defineConfig({
  plugins: [react()],
  server: {
    proxy: {
      '/api': 'http://127.0.0.1:8000',
      '/reports': 'http://127.0.0.1:8000',
    },
  },
})
