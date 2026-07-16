import { memo, useCallback, useEffect, useMemo, useRef, useState } from 'react'
import type { CSSProperties } from 'react'
import { activeBlockAt, clampPlayerHeight, formatTime, hasSpecificTiming, readUrlState, specificTimingRanges, timedBlocks, writeUrlState } from './sync'
import type { JournalBlock, PublicSession, PublicSessionIndex } from './types'
import { useYouTubePlayer } from './useYouTubePlayer'
import './styles.css'

const SESSION_INDEX_URL = '/data/sessions/index.json'

function blockClass(block: JournalBlock, active: boolean) {
  const timingClass = hasSpecificTiming(block) ? 'is-seekable' : block.timing ? 'is-contextual' : 'is-unresolved'
  return [
    'journal-block', `kind-${block.kind}`, timingClass,
    active ? 'is-active' : '',
  ].filter(Boolean).join(' ')
}

const JournalDocument = memo(function JournalDocument({
  pages, activeId, onSelect,
}: {
  pages: PublicSession['pages']
  activeId: string | null
  onSelect: (block: JournalBlock) => void
}) {
  return pages.map((page) => (
    <section className="journal-page" key={page.number} aria-label={`Journal page ${page.number}`}>
      {page.blocks.map((block) => {
        if (block.kind === 'other') return null
        const isActive = activeId === block.id
        const isSeekable = hasSpecificTiming(block)
        const exhibitReference = block.source_time_references.find((reference) => reference.time_domain === 'exhibit')
        const clickSeek = () => {
          const selection = window.getSelection()
          if (selection && !selection.isCollapsed) return
          onSelect(block)
        }
        const keyboardSeek = (event: React.KeyboardEvent<HTMLElement>) => {
          if (event.key !== 'Enter' && event.key !== ' ') return
          event.preventDefault()
          onSelect(block)
        }
        return (
          <article
            id={block.id} key={block.id}
            className={blockClass(block, isActive)}
            aria-current={isActive ? 'true' : undefined}
            role={isSeekable ? 'button' : undefined}
            tabIndex={isSeekable ? 0 : undefined}
            onClick={isSeekable ? clickSeek : undefined}
            onKeyDown={isSeekable ? keyboardSeek : undefined}
          >
            {block.kind === 'heading' ? <h3>{block.text}</h3> : <p>{block.speaker && <strong className="speaker">{block.speaker}. </strong>}{block.text}</p>}
            {exhibitReference && <small className="exhibit-time">Quoted exhibit time {exhibitReference.display ?? formatTime(exhibitReference.time_seconds ?? 0)} · not player time</small>}
          </article>
        )
      })}
    </section>
  ))
})

export default function App() {
  const initialState = useMemo(() => readUrlState(window.location.search), [])
  const [data, setData] = useState<PublicSession | null>(null)
  const [loadError, setLoadError] = useState<string | null>(null)
  const [currentTime, setCurrentTime] = useState(initialState.time ?? 0)
  const [follow, setFollow] = useState(true)
  const [outlineOpen, setOutlineOpen] = useState(false)
  const [playerHeight, setPlayerHeight] = useState(() => clampPlayerHeight(
    window.innerHeight * (window.innerWidth < 740 ? 0.32 : 0.46),
    window.innerHeight,
    window.innerWidth,
  ))
  const programmaticScroll = useRef(false)
  const readerPane = useRef<HTMLElement | null>(null)
  const resizeDrag = useRef<{ pointerId: number; originY: number; originHeight: number } | null>(null)

  useEffect(() => {
    const controller = new AbortController()
    fetch(SESSION_INDEX_URL, { signal: controller.signal })
      .then((response) => {
        if (!response.ok) throw new Error(`Session index returned HTTP ${response.status}`)
        return response.json() as Promise<PublicSessionIndex>
      })
      .then((index) => {
        const requested = new URLSearchParams(window.location.search).get('session')
        const selected = index.sessions.find((item) => item.date === (requested ?? index.latest))
        if (!selected) throw new Error(`Session ${requested} is not published`)
        return fetch(selected.path, { signal: controller.signal })
      })
      .then((response) => {
        if (!response.ok) throw new Error(`Session data returned HTTP ${response.status}`)
        return response.json() as Promise<PublicSession>
      })
      .then(setData)
      .catch((error: unknown) => {
        if (error instanceof DOMException && error.name === 'AbortError') return
        setLoadError(error instanceof Error ? error.message : 'Session data did not load.')
      })
    return () => controller.abort()
  }, [])

  useEffect(() => {
    const clampToViewport = () => setPlayerHeight((height) => clampPlayerHeight(height, window.innerHeight, window.innerWidth))
    window.addEventListener('resize', clampToViewport)
    return () => window.removeEventListener('resize', clampToViewport)
  }, [])

  const allBlocks = useMemo(() => data?.pages.flatMap((page) => page.blocks) ?? [], [data])
  const synchronized = useMemo(() => timedBlocks(allBlocks), [allBlocks])
  const specificallyMapped = useMemo(() => synchronized.filter(hasSpecificTiming), [synchronized])
  const mappedRanges = useMemo(() => specificTimingRanges(synchronized), [synchronized])
  const active = useMemo(() => activeBlockAt(specificallyMapped, currentTime), [specificallyMapped, currentTime])
  const handlePlayerTime = useCallback((seconds: number) => setCurrentTime(seconds), [])
  const player = useYouTubePlayer(data?.sources.video.id ?? '', initialState.time, handlePlayerTime)

  const scrollToBlock = useCallback((blockId: string, behavior: ScrollBehavior = 'smooth', block: ScrollLogicalPosition = 'center') => {
    const element = document.getElementById(blockId)
    if (!element) return
    programmaticScroll.current = true
    element.scrollIntoView({ behavior, block })
    window.setTimeout(() => { programmaticScroll.current = false }, behavior === 'smooth' ? 700 : 50)
  }, [])

  useEffect(() => {
    if (data && initialState.block) window.setTimeout(() => scrollToBlock(initialState.block!, 'auto'), 50)
  }, [data, initialState.block, scrollToBlock])

  useEffect(() => {
    if (follow && active && player.playerState === 1) scrollToBlock(active.id, 'smooth', 'nearest')
  }, [active, follow, player.playerState, scrollToBlock])

  useEffect(() => {
    const pane = readerPane.current
    if (!pane) return
    const interruptFollow = () => {
      if (!programmaticScroll.current) setFollow(false)
    }
    pane.addEventListener('wheel', interruptFollow, { passive: true })
    pane.addEventListener('touchstart', interruptFollow, { passive: true })
    return () => {
      pane.removeEventListener('wheel', interruptFollow)
      pane.removeEventListener('touchstart', interruptFollow)
    }
  }, [data])

  const selectBlock = useCallback((block: JournalBlock) => {
    if (!hasSpecificTiming(block)) return
    setCurrentTime(block.timing.start)
    setFollow(true)
    player.seek(block.timing.start, true)
    window.history.replaceState(null, '', writeUrlState(block.id, block.timing.start))
    scrollToBlock(block.id)
  }, [player, scrollToBlock])

  const startPlayerResize = useCallback((event: React.PointerEvent<HTMLDivElement>) => {
    if (event.button !== 0) return
    event.preventDefault()
    event.currentTarget.setPointerCapture(event.pointerId)
    resizeDrag.current = { pointerId: event.pointerId, originY: event.clientY, originHeight: playerHeight }
  }, [playerHeight])

  const movePlayerResize = useCallback((event: React.PointerEvent<HTMLDivElement>) => {
    const drag = resizeDrag.current
    if (!drag || event.pointerId !== drag.pointerId) return
    event.preventDefault()
    setPlayerHeight(clampPlayerHeight(drag.originHeight + event.clientY - drag.originY, window.innerHeight, window.innerWidth))
  }, [])

  const finishPlayerResize = useCallback((event: React.PointerEvent<HTMLDivElement>) => {
    const drag = resizeDrag.current
    if (!drag || event.pointerId !== drag.pointerId) return
    resizeDrag.current = null
    if (event.currentTarget.hasPointerCapture(event.pointerId)) event.currentTarget.releasePointerCapture(event.pointerId)
  }, [])

  const resizePlayerFromKeyboard = useCallback((event: React.KeyboardEvent<HTMLDivElement>) => {
    if (event.key !== 'ArrowUp' && event.key !== 'ArrowDown') return
    event.preventDefault()
    const direction = event.key === 'ArrowUp' ? -1 : 1
    setPlayerHeight((height) => clampPlayerHeight(height + direction * 24, window.innerHeight, window.innerWidth))
  }, [])

  if (loadError) return (
    <main className="state-page">
      <p className="eyebrow">Session unavailable</p><h1>The official journal data did not load.</h1>
      <p>{loadError}</p><button type="button" onClick={() => window.location.reload()}>Try again</button>
    </main>
  )
  if (!data) return <main className="state-page" aria-live="polite"><p className="eyebrow">Official Journal</p><h1>Preparing the session reader…</h1></main>

  return (
    <div className="app-shell" data-player-ready={player.ready} data-player-time={currentTime.toFixed(1)}>
      <header className="video-stage" style={{ '--player-height': `${playerHeight}px` } as CSSProperties}>
        <div className="player-frame"><div id="youtube-player" aria-label="Official Senate session video" /></div>
        {player.error && <p className="player-error" role="alert">{player.error}</p>}
        <div className="alignment-map" role="img" aria-label={`${specificallyMapped.length} journal passages have direct or reviewed timing. Light segments show their positions in the official session video.`}>
          {mappedRanges.map((range) => {
            const duration = data.sources.video.duration_seconds
            const left = Math.max(0, Math.min(100, range.start / duration * 100))
            const width = Math.max(0, Math.min(100 - left, (range.end - range.start) / duration * 100))
            return <span key={`${range.start}-${range.end}`} style={{ left: `${left}%`, width: `max(2px, ${width}%)` }} />
          })}
        </div>
        <div
          className="stage-resizer" role="separator" aria-label="Resize official video"
          aria-orientation="horizontal" aria-valuemin={96} aria-valuemax={clampPlayerHeight(Number.POSITIVE_INFINITY, window.innerHeight, window.innerWidth)} aria-valuenow={playerHeight}
          tabIndex={0} onPointerDown={startPlayerResize} onPointerMove={movePlayerResize}
          onPointerUp={finishPlayerResize} onPointerCancel={finishPlayerResize}
          onLostPointerCapture={() => { resizeDrag.current = null }} onKeyDown={resizePlayerFromKeyboard}
        ><span aria-hidden="true" /></div>
      </header>

      <section className="reader-pane" ref={readerPane} aria-label="Synchronized official journal">
        <button type="button" className="outline-trigger" onClick={() => setOutlineOpen(true)} aria-label="Open session outline" aria-expanded={outlineOpen}>Outline</button>
        <div className="reading-grid">
        <aside className={outlineOpen ? 'outline is-open' : 'outline'} aria-label="Session outline">
          <button type="button" className="outline-close" onClick={() => setOutlineOpen(false)} aria-label="Close outline">Close</button>
          <ol>
            {data.outline.map((item) => (
              <li key={item.id} className={item.id === active?.section_id ? 'is-current' : ''}>
                <button type="button" onClick={() => {
                  const block = allBlocks.find((candidate) => candidate.id === item.first_block_id)
                  if (block && hasSpecificTiming(block)) selectBlock(block); else scrollToBlock(item.first_block_id)
                  setOutlineOpen(false)
                }}>
                  <span>{item.title}</span>
                </button>
              </li>
            ))}
          </ol>
        </aside>

        <main className="journal" id="journal">
          <JournalDocument
            pages={data.pages}
            activeId={active?.id ?? null}
            onSelect={selectBlock}
          />
          <footer className="record-footer">
            <p>Official Senate Journal · Journal No. {data.session.journal_number} · {new Date(`${data.session.date}T00:00:00`).toLocaleDateString('en-PH', { day: 'numeric', month: 'long', year: 'numeric' })}</p>
            <p className="source-links"><a href={data.sources.journal.url} target="_blank" rel="noreferrer">Official PDF ↗</a><a href={data.sources.video.url} target="_blank" rel="noreferrer">YouTube ↗</a></p>
            <details>
              <summary>About this record</summary>
              <p>{data.processing.alignment_policy}<br /><br />Quoted times inside journal text may belong to exhibits, not the session player.<br /><br />Source: Senate of the Philippines<br />Revision: {data.processing.source_revision}<br />Retrieved: {new Date(data.sources.journal.retrieved_at).toLocaleDateString('en-PH')}<br />Timestamp proposals: {data.processing.alignment_summary.timed_blocks.toLocaleString()} / {data.processing.alignment_summary.total_blocks.toLocaleString()}<br />Review pending: {data.processing.alignment_summary.needs_review.toLocaleString()}<br />Parser: {data.processing.parser}<br />Aligner: {data.processing.aligner ?? 'Pending'}<br />SHA-256: <code>{data.sources.journal.sha256.slice(0, 16)}…</code></p>
            </details>
          </footer>
        </main>
        </div>
      </section>
      {!follow && active && <button type="button" className="return-current" onClick={() => { setFollow(true); scrollToBlock(active.id) }}>Return to current moment · {formatTime(currentTime)}</button>}
    </div>
  )
}
