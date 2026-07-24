export class ApiError extends Error {
  status: number;

  constructor(message: string, status: number) {
    super(message);
    this.status = status;
  }
}

export async function apiFetch(path: string, init: RequestInit = {}): Promise<Response> {
  const response = await fetch(path, { ...init, credentials: "same-origin", cache: "no-store" });
  if (!response.ok) {
    let message = `${response.status} ${response.statusText}`;
    try {
      const payload = await response.json();
      message = payload.detail || payload.error || message;
    } catch {
      // Keep the HTTP status when the body is not JSON.
    }
    throw new ApiError(message, response.status);
  }
  return response;
}

export async function getJson<T>(path: string): Promise<T> {
  return (await apiFetch(path)).json() as Promise<T>;
}

export async function sendJson<T>(path: string, method: string, body?: unknown): Promise<T> {
  return (
    await apiFetch(path, {
      method,
      headers: body === undefined ? undefined : { "Content-Type": "application/json" },
      body: body === undefined ? undefined : JSON.stringify(body),
    })
  ).json() as Promise<T>;
}

export function queryString(values: Record<string, string | number | null | undefined>): string {
  const params = new URLSearchParams();
  Object.entries(values).forEach(([key, value]) => {
    if (value !== undefined && value !== null && value !== "") params.set(key, String(value));
  });
  return params.toString();
}
