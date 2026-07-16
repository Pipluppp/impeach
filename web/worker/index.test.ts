import { afterEach, describe, expect, it, vi } from 'vitest'
import { handleRequest } from './index'

const env = {
  ASSETS: { fetch: vi.fn(() => Promise.resolve(new Response('asset'))) },
  SENATE_RELAY_TOKEN: 'test-relay-token',
}

afterEach(() => vi.unstubAllGlobals())

describe('official Senate relay', () => {
  it('relays the fixed feed route and records the authoritative source', async () => {
    vi.stubGlobal('fetch', vi.fn(() => Promise.resolve(new Response('{"parties":[]}', {
      headers: { 'content-type': 'application/json' },
    }))))
    const response = await handleRequest(
      new Request('https://reader.example/api/official-senate/feed', {
        headers: { authorization: 'Bearer test-relay-token' },
      }),
      env,
    )
    expect(response.status).toBe(200)
    expect(response.headers.get('x-content-source')).toBe(
      'https://senate.gov.ph/hq/impeachment/published',
    )
    expect(await response.json()).toEqual({ parties: [] })
  })

  it('rejects arbitrary proxy targets', async () => {
    const response = await handleRequest(
      new Request('https://reader.example/api/official-senate/document/not-a-pdf', {
        headers: { authorization: 'Bearer test-relay-token' },
      }),
      env,
    )
    expect(response.status).toBe(404)
  })

  it('requires a relay token without exposing it to static assets', async () => {
    const denied = await handleRequest(
      new Request('https://reader.example/api/official-senate/feed'),
      env,
    )
    expect(denied.status).toBe(401)
    const asset = await handleRequest(new Request('https://reader.example/'), env)
    expect(await asset.text()).toBe('asset')
  })
})
