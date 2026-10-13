import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

export default defineConfig({
  plugins: [react()],
  server: {
    port: 5173,
    proxy: {
      // dev talks to the FastAPI process on 8000; `serve` mounts the built
      // bundle itself, so this only matters for `npm run dev`
      '/api': 'http://127.0.0.1:8000',
    },
  },
  build: { outDir: 'dist', emptyOutDir: true },
})
