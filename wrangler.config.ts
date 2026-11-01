import { defineWranglerConfig } from "wrangler/experimental-config";

// Build settings for `cf build` and `cf deploy` (the Wrangler bundler).
// cloudflare.config.ts has no field for the static assets directory.
export default defineWranglerConfig({
  assetsDirectory: "./dist/site",
  types: { generate: false },
});
