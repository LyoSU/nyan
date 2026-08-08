import { mkdtempSync, rmSync } from "node:fs";
import { tmpdir } from "node:os";
import { dirname, join, resolve } from "node:path";
import { fileURLToPath } from "node:url";
import { spawnSync } from "node:child_process";

const scriptDir = dirname(fileURLToPath(import.meta.url));
const projectDir = resolve(scriptDir, "..");
const assetsDir = join(projectDir, "assets");

const scene = join(assetsDir, "u-alive-news-warm-banner-v2.png");
const masthead = join(assetsDir, "u-alive-news-newspaper-masthead.svg");
const output = join(assetsDir, "u-alive-news-scene-banner.png");
const workDir = mkdtempSync(join(tmpdir(), "u-alive-news-banner-"));
const mastheadPng = join(workDir, "masthead.png");
const warpedMasthead = join(workDir, "masthead-warped.png");

function run(command, args) {
  const result = spawnSync(command, args, {
    cwd: projectDir,
    encoding: "utf8",
    stdio: "pipe",
  });

  if (result.status !== 0) {
    const message = result.stderr || result.stdout || `${command} failed`;
    throw new Error(message.trim());
  }
}

try {
  // Keep the branding editable and crisp before it is placed into the scene.
  run("rsvg-convert", [
    "--width",
    "500",
    "--height",
    "230",
    masthead,
    "-o",
    mastheadPng,
  ]);

  // Map the masthead onto the blue panel already printed on the newspaper.
  // Slight transparency lets the original paper/paint texture come through.
  run("magick", [
    mastheadPng,
    "-channel",
    "A",
    "-evaluate",
    "multiply",
    "0.88",
    "+channel",
    "-alpha",
    "set",
    "-virtual-pixel",
    "transparent",
    "-set",
    "option:distort:viewport",
    "1536x1024+0+0",
    "-distort",
    "Perspective",
    "0,0 880,662 500,0 1038,677 0,230 890,738 500,230 1052,758",
    warpedMasthead,
  ]);

  run("magick", [
    scene,
    warpedMasthead,
    "-compose",
    "over",
    "-composite",
    "-depth",
    "8",
    "-strip",
    output,
  ]);

  console.log(output);
} finally {
  rmSync(workDir, { recursive: true, force: true });
}
