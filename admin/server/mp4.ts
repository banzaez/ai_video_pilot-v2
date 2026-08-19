import fs from "node:fs";

function readSlice(fd: number, position: number, length: number): Buffer {
  const buf = Buffer.alloc(length);
  const n = fs.readSync(fd, buf, 0, length, position);
  return n === length ? buf : buf.subarray(0, n);
}

function hasFtyp(head: Buffer): boolean {
  return head.length >= 8 && head.subarray(4, 8).toString("ascii") === "ftyp";
}

function findMoov(fd: number, size: number, head: Buffer): boolean {
  if (head.includes("moov")) return true;
  const tailSize = Math.min(size, 2 * 1024 * 1024);
  const tail = readSlice(fd, size - tailSize, tailSize);
  return tail.includes("moov");
}

/** Незаконченный ffmpeg MP4: ftyp есть, moov ещё не дописан. MPEG-PS (.mp4 с NVR) не трогаем. */
export function isIncompleteMp4(filePath: string): boolean {
  let fd: number | undefined;
  try {
    fd = fs.openSync(filePath, "r");
    const size = fs.fstatSync(fd).size;
    if (size < 64) return true;
    const head = readSlice(fd, 0, Math.min(size, 256 * 1024));
    if (!hasFtyp(head)) return false;
    return !findMoov(fd, size, head);
  } catch {
    return true;
  } finally {
    if (fd != null) {
      try {
        fs.closeSync(fd);
      } catch {
        /* ignore */
      }
    }
  }
}

/** Готовый ISO MP4 (H.264 из convert), не MPEG-PS исходник. */
export function mp4HasMoov(filePath: string): boolean {
  let fd: number | undefined;
  try {
    fd = fs.openSync(filePath, "r");
    const size = fs.fstatSync(fd).size;
    if (size < 64) return false;
    const head = readSlice(fd, 0, Math.min(size, 256 * 1024));
    if (!hasFtyp(head)) return false;
    return findMoov(fd, size, head);
  } catch {
    return false;
  } finally {
    if (fd != null) {
      try {
        fs.closeSync(fd);
      } catch {
        /* ignore */
      }
    }
  }
}
