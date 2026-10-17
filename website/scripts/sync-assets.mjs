import { cp, rm } from "node:fs/promises";

await rm("public/assets", { recursive: true, force: true });

await cp("../assets", "public/assets", {
  recursive: true,
});

console.log("Synced repository assets.");