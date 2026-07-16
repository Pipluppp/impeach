import { useCallback, useEffect, useMemo, useRef, useState } from 'react'
import type { YouTubePlayer } from './types'

type PlayerEvent = { target: YouTubePlayer }
type StateEvent = PlayerEvent & { data: number }
type ErrorEvent = { data: number }

export function useYouTubePlayer(videoId: string, initialTime: number | null, onTime: (seconds: number) => void) {
  const playerRef = useRef<YouTubePlayer | null>(null)
  const pendingSeekRef = useRef<{ seconds: number; play: boolean } | null>(null)
  const readyRef = useRef(false)
  const onTimeRef = useRef(onTime)
  const [ready, setReady] = useState(false)
  const [playerState, setPlayerState] = useState(-1)
  const [error, setError] = useState<string | null>(null)
  useEffect(() => { onTimeRef.current = onTime }, [onTime])

  useEffect(() => {
    if (!videoId) return
    let disposed = false
    let interval: number | undefined

    const createPlayer = () => {
      if (disposed || !window.YT || playerRef.current) return
      playerRef.current = new window.YT.Player('youtube-player', {
        videoId,
        width: '100%',
        height: '100%',
        playerVars: { playsinline: 1, rel: 0, origin: window.location.origin },
        events: {
          onReady: (event: PlayerEvent) => {
            readyRef.current = true
            const pending = pendingSeekRef.current
            if (pending) {
              event.target.seekTo(pending.seconds, true)
              if (pending.play) event.target.playVideo()
              pendingSeekRef.current = null
            } else if (initialTime !== null) event.target.seekTo(initialTime, true)
            setReady(true)
            interval = window.setInterval(() => {
              const player = playerRef.current
              if (player) onTimeRef.current(player.getCurrentTime())
            }, 100)
          },
          onStateChange: (event: StateEvent) => setPlayerState(event.data),
          onError: (event: ErrorEvent) => setError(`Official video could not be loaded (YouTube error ${event.data}).`),
        },
      })
    }

    if (window.YT?.Player) createPlayer()
    else {
      const previous = window.onYouTubeIframeAPIReady
      window.onYouTubeIframeAPIReady = () => {
        previous?.()
        createPlayer()
      }
      if (!document.querySelector('script[data-youtube-api]')) {
        const script = document.createElement('script')
        script.src = 'https://www.youtube.com/iframe_api'
        script.async = true
        script.dataset.youtubeApi = 'true'
        script.onerror = () => setError('The YouTube player API did not load. The official video link remains available.')
        document.head.append(script)
      }
    }

    return () => {
      disposed = true
      if (interval !== undefined) window.clearInterval(interval)
      playerRef.current?.destroy()
      playerRef.current = null
      readyRef.current = false
      pendingSeekRef.current = null
    }
  }, [initialTime, videoId])

  const seek = useCallback((seconds: number, play = true) => {
    const player = playerRef.current
    if (!player || !readyRef.current) {
      pendingSeekRef.current = { seconds, play }
      onTimeRef.current(seconds)
      return false
    }
    player.seekTo(seconds, true)
    if (play) player.playVideo()
    onTimeRef.current(seconds)
    return true
  }, [])

  return useMemo(
    () => ({ ready, playerState, error, seek }),
    [error, playerState, ready, seek],
  )
}
