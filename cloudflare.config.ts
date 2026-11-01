import { defineConfig } from "cf/config";

// Static assets only: no Worker code runs. The directory that is served, dist/site,
// is a build setting and lives in wrangler.config.ts (`assetsDirectory`).
// Build it first with ./scripts/build-digital-release.sh (or `npm run build:site`).
export default defineConfig({
  worker: {
    name: "open-llm",
    compatibilityDate: "2026-10-23",
    assets: {
      htmlHandling: "auto-trailing-slash",
      notFoundHandling: "404-page",
    },
  },
});
