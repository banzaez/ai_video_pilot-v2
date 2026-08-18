import path from "node:path";
import { fileURLToPath } from "node:url";

const serverDir = path.dirname(fileURLToPath(import.meta.url));
export const adminDir = path.resolve(serverDir, "..");
export const videoDir = path.resolve(adminDir, "../data/video");
export const resultsDir = path.resolve(adminDir, "../data/results");
export const mapsDir = path.resolve(adminDir, "../data/maps");
export const camerasDir = path.join(mapsDir, "cameras");
export const projectRoot = path.resolve(adminDir, "..");
