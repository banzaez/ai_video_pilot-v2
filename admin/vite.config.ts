import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";
import { adminDir, resultsDir, videoDir } from "./server/config.js";
import { mediaLibraryPlugin } from "./server/plugin.js";

export default defineConfig({
  plugins: [react(), mediaLibraryPlugin()],
  server: {
    port: 5173,
    host: "localhost",
    fs: {
      allow: [adminDir, videoDir, resultsDir],
    },
    proxy: {
      "/api/jobs": {
        target: "http://127.0.0.1:8765",
        changeOrigin: true,
      },
    },
  },
});
