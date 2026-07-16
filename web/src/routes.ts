const SESSION_ROUTE = /^\/sessions\/(\d{4}-\d{2}-\d{2})\/?$/

export function sessionDateFromPath(pathname: string): string | null {
  return SESSION_ROUTE.exec(pathname)?.[1] ?? null
}

export function sessionHref(date: string): string {
  return `/sessions/${date}`
}

export function sessionLinkLabel(date: string): string {
  const label = new Date(`${date}T00:00:00`).toLocaleDateString('en-PH', {
    day: 'numeric',
    month: 'long',
    year: 'numeric',
  })
  return `${label} Impeachment Trial`
}
