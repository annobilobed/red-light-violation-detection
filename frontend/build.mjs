// Bundles the dashboard into backend/app/static (served by FastAPI).
import * as esbuild from "esbuild";
import { copyFileSync, mkdirSync } from "node:fs";

const out = "../backend/app/static";
mkdirSync(out, { recursive: true });
copyFileSync("public/index.html", `${out}/index.html`);
const opts = {
  entryPoints: { app: "src/main.jsx" },
  bundle: true,
  minify: true,
  sourcemap: false,
  outdir: out,
  loader: { ".jsx": "jsx" },
  jsx: "automatic",
  define: { "process.env.NODE_ENV": '"production"' },
  target: ["es2020"],
  logLevel: "info",
};
if (process.argv.includes("--watch")) {
  const ctx = await esbuild.context(opts);
  await ctx.watch();
} else {
  await esbuild.build(opts);
}
