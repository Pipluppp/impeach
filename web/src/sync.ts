import type { JournalBlock } from './types'

export type TimedBlock = JournalBlock & { timing: NonNullable<JournalBlock['timing']> }
export type TimelineRange = { start: number; end: number }

export function timedBlocks(blocks: JournalBlock[]): TimedBlock[] {
  return blocks
    .filter((block): block is TimedBlock => block.timing !== null)
    .sort((a, b) => a.timing.start - b.timing.start || a.sequence - b.sequence)
}

export function hasSpecificTiming(block: JournalBlock): block is TimedBlock {
  if (!block.timing) return false
  return block.timing.review_state === 'manual_reviewed' ||
    block.timing.precision === 'approximate_dialogue_turn'
}

export function specificTimingRanges(blocks: JournalBlock[], mergeGapSeconds = 1): TimelineRange[] {
  const ranges: TimelineRange[] = []
  for (const block of blocks) {
    if (hasSpecificTiming(block) && block.timing.end > block.timing.start) {
      ranges.push({ start: block.timing.start, end: block.timing.end })
    }
  }
  ranges.sort((a, b) => a.start - b.start || a.end - b.end)

  const merged: TimelineRange[] = []
  for (const range of ranges) {
    const previous = merged.at(-1)
    if (previous && range.start <= previous.end + mergeGapSeconds) previous.end = Math.max(previous.end, range.end)
    else merged.push({ ...range })
  }
  return merged
}

const precisionRank = (precision: string) =>
  precision.includes('dialogue') ? 3 : precision.includes('section') ? 1 : 2

const PLAYER_TIME_EPSILON_SECONDS = 0.25

export function activeBlockAt(blocks: TimedBlock[], seconds: number): TimedBlock | null {
  let low = 0
  let high = blocks.length - 1
  let lastStart = -1
  while (low <= high) {
    const middle = (low + high) >> 1
    if (blocks[middle].timing.start <= seconds + PLAYER_TIME_EPSILON_SECONDS) {
      lastStart = middle
      low = middle + 1
    } else high = middle - 1
  }
  if (lastStart < 0) return null
  const candidates: TimedBlock[] = []
  for (let index = lastStart; index >= 0 && blocks[index].timing.start >= seconds - 1800; index -= 1) {
    if (blocks[index].timing.end >= seconds - PLAYER_TIME_EPSILON_SECONDS) candidates.push(blocks[index])
  }
  return candidates.sort((a, b) =>
    precisionRank(b.timing.precision) - precisionRank(a.timing.precision) ||
    b.timing.start - a.timing.start || a.sequence - b.sequence
  )[0] ?? null
}

export function readUrlState(search: string): { block: string | null; time: number | null } {
  const params = new URLSearchParams(search)
  const value = params.get('t')
  const parsed = value === null ? null : Number(value)
  return { block: params.get('block'), time: parsed !== null && Number.isFinite(parsed) && parsed >= 0 ? parsed : null }
}

export function writeUrlState(block: string, time: number): string {
  const params = new URLSearchParams(window.location.search)
  params.set('block', block)
  params.set('t', time.toFixed(1))
  return `${window.location.pathname}?${params.toString()}${window.location.hash}`
}

export function formatTime(seconds: number): string {
  const rounded = Math.max(0, Math.floor(seconds))
  const hours = Math.floor(rounded / 3600)
  const minutes = Math.floor((rounded % 3600) / 60)
  const rest = rounded % 60
  return hours > 0
    ? `${hours}:${String(minutes).padStart(2, '0')}:${String(rest).padStart(2, '0')}`
    : `${minutes}:${String(rest).padStart(2, '0')}`
}

export function clampPlayerHeight(value: number, viewportHeight: number, viewportWidth = Number.POSITIVE_INFINITY): number {
  const maximum = Math.max(96, Math.min(viewportHeight * 0.85, viewportWidth * 9 / 16))
  return Math.round(Math.min(maximum, Math.max(96, value)))
}
