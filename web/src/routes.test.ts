import { describe, expect, it } from 'vitest'
import { sessionDateFromPath, sessionHref, sessionLinkLabel } from './routes'

describe('session routes', () => {
  it('recognizes canonical dated reader routes', () => {
    expect(sessionDateFromPath('/sessions/2026-07-13')).toBe('2026-07-13')
    expect(sessionDateFromPath('/sessions/2026-07-13/')).toBe('2026-07-13')
    expect(sessionDateFromPath('/')).toBeNull()
    expect(sessionDateFromPath('/sessions/not-a-date')).toBeNull()
  })

  it('builds plain session links', () => {
    expect(sessionHref('2026-07-14')).toBe('/sessions/2026-07-14')
    expect(sessionLinkLabel('2026-07-14')).toBe('July 14, 2026 Impeachment Trial')
  })
})
