export type EntitySpace = "t" | "g" | "p";

export type EntityId = {
  space: EntitySpace;
  n: number;
};

const SPACES = new Set<EntitySpace>(["t", "g", "p"]);
const TOKEN = /^([tgp])(\d+)$/i;

export function trackletId(n: number): EntityId {
  return { space: "t", n };
}

export function groupId(n: number): EntityId {
  return { space: "g", n };
}

export function personId(n: number): EntityId {
  return { space: "p", n };
}

export function formatEntityId(id: EntityId): string {
  return `${id.space}${id.n}`;
}

export function parseEntityId(raw: string): EntityId {
  const text = raw.trim();
  const m = TOKEN.exec(text);
  if (!m) {
    throw new Error(`Ожидался EntityId вида t1/g1/p1, получено ${JSON.stringify(raw)}`);
  }
  const space = m[1]!.toLowerCase() as EntitySpace;
  const n = Number(m[2]);
  if (!SPACES.has(space) || !Number.isInteger(n) || n <= 0) {
    throw new Error(`Ожидался EntityId вида t1/g1/p1, получено ${JSON.stringify(raw)}`);
  }
  return { space, n };
}

export function parseEntityIdOptional(raw: unknown): EntityId | null {
  if (typeof raw !== "string" || !raw.trim()) return null;
  try {
    return parseEntityId(raw);
  } catch {
    return null;
  }
}

/** Ключ в словаре лиц: `g1`, иначе легаси-голое `"1"` как группа. */
export function faceBucketKeys(id: EntityId): string[] {
  const prefixed = formatEntityId(id);
  return [prefixed, String(id.n)];
}
