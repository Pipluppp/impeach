export type Timing = {
  start: number
  end: number
  time_domain: 'session_video'
  precision: string
  confidence: number
  review_state: string
}

export type TimeReference = {
  time_domain: string
  time_seconds?: number
  time_seconds_since_midnight?: number
  display?: string
  exhibit_id?: string
}

export type JournalBlock = {
  id: string
  sequence: number
  kind: string
  text: string
  speaker: string | null
  section_id: string | null
  role: string | null
  timing: Timing | null
  source_time_references: TimeReference[]
}

export type OutlineItem = {
  id: string
  title: string
  first_block_id: string
  page: number
  timing: Timing | null
}

export type PublicSession = {
  schema_version: '1.1.0'
  session: { id: string; date: string; journal_number: number; title: string }
  sources: {
    journal: { url: string; listing_url: string; retrieved_at: string; sha256: string; page_count: number }
    video: { id: string; url: string; playlist_id: string; channel: string; duration_seconds: number }
  }
  processing: {
    source_revision: number
    parser: string
    aligner: string | null
    alignment_summary: {
      total_blocks: number
      timed_blocks: number
      coverage: number
      needs_review: number
      manual_reviewed: number
      unresolved_conflicts: number
    }
    alignment_policy: string
  }
  outline: OutlineItem[]
  pages: Array<{ number: number; blocks: JournalBlock[] }>
}

export type PublicSessionIndex = {
  schema_version: 1
  latest: string
  sessions: Array<{ date: string; journal_number: number; title: string; path: string }>
}

export type YouTubePlayer = {
  seekTo(seconds: number, allowSeekAhead: boolean): void
  playVideo(): void
  getCurrentTime(): number
  getPlayerState(): number
  destroy(): void
}

export type YouTubeNamespace = {
  Player: new (elementId: string, options: Record<string, unknown>) => YouTubePlayer
  PlayerState: { PLAYING: number; BUFFERING: number }
}

declare global {
  interface Window {
    YT?: YouTubeNamespace
    onYouTubeIframeAPIReady?: () => void
  }
}
