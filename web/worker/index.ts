const SENATE_ORIGIN = 'https://senate.gov.ph'
const FEED_PATH = '/hq/impeachment/published'
const DOCUMENT_RE = /^\/api\/official-senate\/document\/([0-9a-f-]+\.pdf)$/i
const BROWSER_USER_AGENT = 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) '
  + 'AppleWebKit/537.36 (KHTML, like Gecko) Chrome/140.0.0.0 Safari/537.36'

interface Env {
  ASSETS: { fetch(request: Request): Promise<Response> }
  SENATE_RELAY_TOKEN: string
}

const errorResponse = (status: number, message: string) => Response.json(
  { error: message },
  { status, headers: { 'cache-control': 'no-store' } },
)

const authorized = async (request: Request, expectedToken: string) => {
  const supplied = request.headers.get('authorization')?.replace(/^Bearer\s+/i, '') ?? ''
  const encoder = new TextEncoder()
  const [expectedHash, suppliedHash] = await Promise.all([
    crypto.subtle.digest('SHA-256', encoder.encode(expectedToken)),
    crypto.subtle.digest('SHA-256', encoder.encode(supplied)),
  ])
  const expected = new Uint8Array(expectedHash)
  const actual = new Uint8Array(suppliedHash)
  let difference = 0
  for (let index = 0; index < expected.length; index += 1) {
    difference |= expected[index] ^ actual[index]
  }
  return difference === 0 && supplied.length > 0
}

const relay = async (request: Request, upstreamPath: string, cacheControl: string) => {
  if (request.method !== 'GET' && request.method !== 'HEAD') {
    return errorResponse(405, 'method not allowed')
  }
  const upstream = await fetch(`${SENATE_ORIGIN}${upstreamPath}`, {
    method: request.method,
    headers: {
      accept: upstreamPath.endsWith('.pdf') ? 'application/pdf,*/*' : 'application/json,*/*',
      'accept-language': 'en-US,en;q=0.9',
      referer: `${SENATE_ORIGIN}/services/impeachment-documents`,
      'user-agent': BROWSER_USER_AGENT,
    },
    redirect: 'follow',
  })
  if (!upstream.ok || !upstream.body) {
    return errorResponse(502, `official Senate source returned HTTP ${upstream.status}`)
  }
  const headers = new Headers()
  for (const name of ['content-type', 'content-length', 'etag', 'last-modified']) {
    const value = upstream.headers.get(name)
    if (value) headers.set(name, value)
  }
  headers.set('cache-control', cacheControl)
  headers.set('x-content-source', `${SENATE_ORIGIN}${upstreamPath}`)
  return new Response(upstream.body, { status: 200, headers })
}

export const handleRequest = async (request: Request, env: Env): Promise<Response> => {
  const { pathname } = new URL(request.url)
  if (
    pathname.startsWith('/api/official-senate/')
    && !await authorized(request, env.SENATE_RELAY_TOKEN)
  ) {
    return errorResponse(401, 'unauthorized')
  }
  if (pathname === '/api/official-senate/feed') {
    return relay(request, FEED_PATH, 'public, max-age=300')
  }
  const document = DOCUMENT_RE.exec(pathname)
  if (document) {
    return relay(
      request,
      `/hq/uploads/impeachment/${document[1]}`,
      'public, max-age=86400, immutable',
    )
  }
  if (pathname.startsWith('/api/official-senate/')) {
    return errorResponse(404, 'unknown official-source route')
  }
  return env.ASSETS.fetch(request)
}

export default { fetch: handleRequest } satisfies ExportedHandler<Env>
