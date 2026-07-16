import { describe, expect, it } from 'vitest'
import { activeBlockAt, clampPlayerHeight, formatTime, hasSpecificTiming, readUrlState, specificTimingRanges, timedBlocks } from './sync'
import type { JournalBlock } from './types'

const block = (id: string, sequence: number, start: number, end: number, precision: string, reviewState = 'auto_accepted'): JournalBlock => ({
  id, sequence, kind: 'speaker_utterance', text: id, speaker: null, section_id: null, role: null,
  source_time_references: [], timing: { start, end, time_domain: 'session_video', precision, confidence: 0.8, review_state: reviewState },
})

describe('synchronization helpers', () => {
  it('selects the dialogue turn over an overlapping summary', () => {
    const blocks = timedBlocks([
      block('summary', 1, 10, 40, 'narrative_summary_range'),
      block('dialogue', 2, 20, 25, 'approximate_dialogue_turn'),
    ])
    expect(activeBlockAt(blocks, 22)?.id).toBe('dialogue')
    expect(activeBlockAt(blocks, 35)?.id).toBe('summary')
    expect(activeBlockAt(blocks, 45)).toBeNull()
    expect(activeBlockAt(blocks, 19.9)?.id).toBe('dialogue')
  })

  it('parses share state defensively', () => {
    expect(readUrlState('?block=j1&t=721.4')).toEqual({ block: 'j1', time: 721.4 })
    expect(readUrlState('?t=-4')).toEqual({ block: null, time: null })
  })

  it('formats long session time', () => expect(formatTime(20776)).toBe('5:46:16'))

  it('seeks direct dialogue proposals and manually reviewed timing, but not contextual proposals', () => {
    expect(hasSpecificTiming(block('accepted-dialogue', 1, 10, 12, 'approximate_dialogue_turn'))).toBe(true)
    expect(hasSpecificTiming(block('pending-dialogue', 2, 20, 22, 'approximate_dialogue_turn', 'needs_review'))).toBe(true)
    expect(hasSpecificTiming(block('accepted-summary', 3, 30, 40, 'narrative_summary_range'))).toBe(false)
    expect(hasSpecificTiming(block('reviewed-summary', 4, 50, 60, 'narrative_summary_range', 'manual_reviewed'))).toBe(true)
  })

  it('merges nearby specific timing into compact timeline ranges', () => {
    expect(specificTimingRanges([
      block('first', 1, 10, 12, 'approximate_dialogue_turn'),
      block('nearby', 2, 12.5, 15, 'approximate_dialogue_turn'),
      block('direct-pending', 3, 17, 19, 'approximate_dialogue_turn', 'needs_review'),
      block('unclear', 4, 18, 24, 'contextual_dialogue_range', 'needs_review'),
      block('reviewed', 5, 30, 32, 'narrative_summary_range', 'manual_reviewed'),
    ])).toEqual([{ start: 10, end: 15 }, { start: 17, end: 19 }, { start: 30, end: 32 }])
  })

  it('keeps a resized player inside the readable viewport budget', () => {
    expect(clampPlayerHeight(40, 900)).toBe(96)
    expect(clampPlayerHeight(120, 900)).toBe(120)
    expect(clampPlayerHeight(420, 900)).toBe(420)
    expect(clampPlayerHeight(900, 900)).toBe(765)
    expect(clampPlayerHeight(500, 844, 390)).toBe(219)
  })
})
