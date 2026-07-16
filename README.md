# Senate synchronized journal reader

The official Philippine Senate impeachment journal is the public text. Timestamped ASR is an
internal aid used to map journal blocks to the official Senate YouTube session; narrative summaries
are labeled as approximate ranges, never as word-level transcript timing.

The repository has two product seams:

```text
.github/workflows/  scheduled checks, processing, validation, deployment
pipeline/           discover → preserve PDF → parse → transcribe → align → publish JSON
data/               Git-backed source PDFs, metadata, Markdown, ASR, and alignments
web/                React reader and the generated static session payload
```

## Current vertical slice

Journal No. 6 (14 July 2026) is paired with official video `GrQeE6SB1YY`. The tracked record
contains the source PDF and provenance, 92-page Markdown, 2,405 structured journal blocks, 4,710
ASR segments, 2,105 bounded timestamp ranges, and the schema-validated browser payload. Eleven
ranges are human-reviewed and 598 distinctive monotonic matches are auto-accepted; another 428
direct ASR proposals remain review-pending but are usable as approximate seek targets. The 1,068
neighbor-bounded contextual proposals remain non-interactive, and another 847 spoken blocks are
deliberately unaligned. Journal display text and normalized matching text are stored separately.

Alignment is a deep module behind one `align(journal, transcript, overrides)` interface. It keeps
printed speaker labels and structural headings out of spoken-text queries, tolerates fuzzy and
joined/split ASR words, selects a globally monotonic candidate path, and abstains when evidence is
insufficient. With manual overrides removed, the preserved navigation benchmark improved from
4/12 to 11/12 targets within ten seconds; the remaining greeting is unintelligible in the ASR.

The web reader embeds YouTube, seeks from reviewed or accepted journal blocks, follows playback,
provides a session outline, keeps source details at the end of the record, and preserves `block`
and `t` in the URL. It is already
React 19 + TypeScript + Vite. React is useful for player, URL, selection, follow, drawer, and error
state; a heavier framework or server-side runtime is not needed.

Cloudflare serves `web/dist` as Workers Static Assets. The session JSON is built from this Git
repository into `web/public/data`, so R2, D1, Durable Objects, and hosted media are not MVP
dependencies. Only a Cloudflare API token/account ID is needed when an approved deployment runs.

## Automation status

`refresh.yml` checks the official Senate feed and official YouTube playlist twice daily, away from
the top of the hour. It dispatches only a newly dated or revised, uniquely matched session.
`process.yml` then preserves and parses the PDF, acquires temporary audio, plans overlapping
30-minute chunks, runs up to 18 GitHub-hosted CPU ASR jobs, merges their absolute session-video
timestamps, aligns the journal, validates the web build, and optionally opens a review PR. Audio,
models, and chunks have one-day artifact/cache lifetimes and are never committed.

The orchestration lives behind `pipeline/automation.py`; workflow YAML only supplies hosted-runner
plumbing. Local validation covers the real 92-page PDF twice with identical output, a real
60-second `whisper.cpp v1.8.6` transcription at 0.16× realtime, deterministic fan-in and full
2,405-block finalize idempotency, all Python tests, workflow linting, and the React/Cloudflare
build. A hosted run still requires an approved
push and manual workflow dispatch. The official Senate site currently returns Cloudflare 403 from
this development network even in headless Chrome, so official-source retrieval is bounded and
fails without replacing prior records; the GitHub runner route must be observed in that first run.

Fan-out reduces elapsed time but not billed runner minutes. GitHub Free permits 20 concurrent
standard jobs account-wide; this workflow reserves two slots by capping itself at 18. Public
repositories receive standard hosted runners without minute charges, while a private repository's
included minutes remain the practical constraint.

## Run and verify

```powershell
python -m pip install -r pipeline/requirements-dev.txt
python -m pytest -q pipeline/tests

Push-Location web
npm ci
npm run check
npx wrangler deploy --dry-run --env preview
npm run dev
Pop-Location
```

Rebuild the public payload after alignment changes:

```powershell
python pipeline/build_public_data.py `
  --session data/sessions/2026-07-14/session.json `
  --source data/sessions/2026-07-14/source/source.json `
  --journal data/sessions/2026-07-14/journal/blocks.json `
  --alignment data/sessions/2026-07-14/alignment/alignments.json `
  --output web/public/data/sessions/2026-07-14.json
```

Material remaining risks are the Senate endpoint's intermittent Cloudflare 403 response, changing
YouTube extraction behavior, the unresolved PyMuPDF distribution-license choice, and the large
human review queue. Local experiment history, screenshots, benchmark dumps, and the supplied task
pack are intentionally ignored rather than carried as repository surface area.
