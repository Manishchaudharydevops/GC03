# Groundcheck

A working RAG prototype: describe a startup idea, get a verdict grounded in
community evidence, with every claim traceable back to a specific cited post.

This is a **real, tested pipeline** — retrieval + LLM synthesis + a UI that
renders structured citations — running on a **synthetic demo dataset** of 30
Reddit-style posts (`api/data/posts.json`), because live Reddit access
requires a paid commercial API agreement (Reddit's official commercial tier
starts around $12,000/year) or a third-party data provider. Swapping in real
data is a small, isolated change — see "Going live with real data" below.

## Stack

- **Frontend**: plain HTML/CSS/JS at the project root — no build step, no
  framework. Served directly by Vercel's static hosting.
- **Backend**: a single Node.js serverless function at `api/ask.js`. This is
  Vercel's default, zero-config runtime — a file in `/api` exporting a
  handler automatically becomes an endpoint at `/api/ask`. No extra config
  file, no entrypoint declaration needed.
- **Retrieval**: hand-rolled TF-IDF + cosine similarity over the demo
  dataset — no external embedding API, no vector DB, fast cold starts.
- **Generation**: Claude (`@anthropic-ai/sdk`), given only the retrieved
  posts, forced to output structured JSON and forbidden from inventing
  quotes or claims not present in the evidence.

## Project structure

```
groundcheck/
├── index.html
├── styles.css
├── script.js
├── api/
│   ├── ask.js            ← POST /api/ask — retrieval + Claude synthesis
│   └── data/posts.json   ← demo dataset
├── package.json
└── .gitignore
```

## Run locally

```bash
npm install
npm i -g vercel
export ANTHROPIC_API_KEY=sk-ant-...
vercel dev
```

Open the printed local URL (usually http://localhost:3000) and submit an idea.

## Deploy to Vercel

```bash
vercel
```

Then, in the Vercel dashboard → your project → Settings → Environment
Variables, add:

- `ANTHROPIC_API_KEY` — your Anthropic API key

Redeploy after adding it (environment variable changes require a new
deployment to take effect). No `vercel.json` is needed — this is a
zero-config deployment.

## Going live with real data

Everything downstream of retrieval is already source-agnostic. To point this
at real data instead of the demo dataset:

1. Replace the `require('./data/posts.json')` line in `api/ask.js` with a
   call to your data source — Reddit's official Data API (commercial tier
   required for production use), or a third-party provider (Data365, Apify
   actors, etc.).
2. Keep the same post shape: `{id, subreddit, title, text, score, author, url}`.
3. Consider caching retrieval results (e.g. Vercel KV, or Postgres +
   pgvector) once you're pulling from a live source, so you're not
   re-fetching and re-scoring on every request.
4. If your corpus grows past a few thousand posts, swap the hand-rolled
   TF-IDF for real embeddings (OpenAI/Voyage) + a vector DB — the TF-IDF
   approach here is intentionally lightweight for a small demo corpus, not
   built to scale to tens of thousands of documents.

## Notes on the current build

- The dataset is entirely synthetic — written to be representative of real
  discussion patterns, not scraped or copied from anywhere. It will only
  "know about" the ~30 topics seeded in `api/data/posts.json`. Ideas far
  outside those topics will get a low score or an "Insufficient Evidence"
  verdict — that's the system working correctly, not a bug.
- The system prompt explicitly disallows quoting verbatim and requires every
  claim to cite a post id.
- This was switched from an earlier Python/Vercel build after hitting two
  rounds of undocumented quirks in Vercel's newer Python runtime (entrypoint
  detection, and static files being swallowed by the Python app once an
  entrypoint was declared). The Node.js runtime used here is Vercel's
  original, zero-config serverless function format and has none of those
  issues — every request/response path in `api/ask.js` was tested locally
  end to end (success path blocked only by the missing API key, GET
  rejection, empty-input validation) before this was handed to you.
