import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

// Builds into ../static so FastAPI's existing StaticFiles mount (app/main.py)
// serves the compiled SPA with zero backend changes. Dev server proxies
// /api to the FastAPI process instead of needing CORS there.
export default defineConfig({
  plugins: [react()],
  server: {
    proxy: {
      "/api": "http://127.0.0.1:8000",
    },
  },
  build: {
    outDir: "../static",
    emptyOutDir: true,
  },
});
