import { cp, readdir, rm } from "node:fs/promises";

const RASTER = /\.(png|jpe?g|gif|bmp|tiff?)$/i;

const rasters = (await readdir("../assets", { recursive: true })).filter((file) => RASTER.test(file));
if (rasters.length > 0) {
  console.warn(
    `Raster assets should be WebP; run \`python scripts/convert_assets_to_webp.py\`:\n${rasters.map((file) => `  assets/${file}`).join("\n")}`,
  );
}

await rm("public/assets", { recursive: true, force: true });

await cp("../assets", "public/assets", {
  recursive: true,
});

console.log("Synced repository assets.");
