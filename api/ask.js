/**
 * Groundcheck RAG endpoint — Node.js runtime (Vercel's default, zero-config
 * serverless function format: a file in /api exporting a default handler).
 *
 * Retrieval: lightweight hand-rolled TF-IDF + cosine similarity over the demo
 * dataset (api/data/posts.json). No heavy deps -> fast cold starts.
 *
 * Generation: Claude synthesizes a grounded verdict, citing only the
 * retrieved posts by id. The prompt forces JSON-only output.
 *
 * Swap-in path for real data: replace loadPosts() with a call to your Reddit
 * data source (official API, or a third-party provider -- see README) and
 * keep everything downstream the same.
 */

const Anthropic = require('@anthropic-ai/sdk');
const posts = require('./data/posts.json');

const TOP_K = 6;
const STOPWORDS = new Set([
  'the', 'a', 'an', 'and', 'or', 'but', 'is', 'are', 'was', 'were', 'be',
  'to', 'of', 'in', 'on', 'for', 'with', 'that', 'this', 'it', 'i', 'you',
  'my', 'me', 'at', 'as', 'if', 'would', 'will', 'just', 'so', 'not',
  'have', 'has', 'had', 'do', 'does', 'did', 'can', 'could', 'than',
  'then', 'into', 'over', 'your', 'their', 'them', 'from', 'by', 'about',
]);

function tokenize(text) {
  const words = (text.toLowerCase().match(/[a-z']+/g)) || [];
  return words.filter((w) => !STOPWORDS.has(w) && w.length > 2);
}

function buildCorpus(allPosts) {
  const docs = allPosts.map((p) => tokenize(p.title + ' ' + p.text));
  const df = new Map();
  docs.forEach((doc) => {
    new Set(doc).forEach((term) => df.set(term, (df.get(term) || 0) + 1));
  });
  const nDocs = docs.length;
  const idf = new Map();
  df.forEach((freq, term) => idf.set(term, Math.log((nDocs + 1) / (freq + 1)) + 1));

  const vectorize = (tokens) => {
    const tf = new Map();
    tokens.forEach((t) => tf.set(t, (tf.get(t) || 0) + 1));
    const vec = new Map();
    tf.forEach((count, term) => vec.set(term, count * (idf.get(term) || 0)));
    return vec;
  };

  const docVecs = docs.map(vectorize);
  return { docVecs, vectorize };
}

function cosine(vecA, vecB) {
  let dot = 0;
  vecA.forEach((val, term) => {
    if (vecB.has(term)) dot += val * vecB.get(term);
  });
  const normA = Math.sqrt([...vecA.values()].reduce((s, v) => s + v * v, 0)) || 1e-9;
  const normB = Math.sqrt([...vecB.values()].reduce((s, v) => s + v * v, 0)) || 1e-9;
  return dot / (normA * normB);
}

function retrieve(query, allPosts, docVecs, vectorize, k = TOP_K) {
  const qVec = vectorize(tokenize(query));
  const scored = allPosts.map((_, i) => [cosine(qVec, docVecs[i]), i]);
  scored.sort((a, b) => b[0] - a[0]);
  let results = scored.slice(0, k).filter(([score]) => score > 0).map(([, i]) => allPosts[i]);
  if (results.length === 0) results = allPosts.slice(0, k);
  return results;
}

const SYSTEM_PROMPT = `You are Groundcheck's analysis engine. You validate startup ideas using ONLY the community evidence provided to you -- never your own opinion of the market, and never facts outside the given evidence.

Rules:
- Every claim must trace back to one of the provided post ids.
- Never invent a quote. Paraphrase in your own words; do not copy sentences verbatim.
- If the evidence is thin or off-topic for the idea, say so plainly instead of forcing a confident verdict.
- Output ONLY valid JSON, no markdown fences, no preamble, matching this exact shape:

{
  "verdict_score": <integer 1-10>,
  "verdict_label": "<one of: Strong Signal, Mixed Signal, Weak Signal, Insufficient Evidence>",
  "one_line_verdict": "<one sentence, plain language>",
  "sentiment": {"positive": <0-100 int>, "neutral": <0-100 int>, "negative": <0-100 int>},
  "themes": [
    {"theme": "<short theme name>", "summary": "<1-2 sentence paraphrase>", "post_ids": ["p001", "p002"]}
  ],
  "citations": [
    {"post_id": "p001", "paraphrase": "<one sentence, your own words, no quotes>"}
  ],
  "recommendation": "<2-3 sentences of concrete next step advice grounded in the evidence>"
}`;

async function callClaude(idea, evidencePosts) {
  const client = new Anthropic(); // reads ANTHROPIC_API_KEY from env
  const evidenceBlock = evidencePosts
    .map((p) => `id: ${p.id}\nsubreddit: ${p.subreddit}\ntitle: ${p.title}\ntext: ${p.text}\nupvotes: ${p.score}`)
    .join('\n\n');

  const userPrompt = `Startup idea to validate: "${idea}"\n\nCommunity evidence (demo dataset):\n${evidenceBlock}\n\nReturn the JSON object now.`;

  const response = await client.messages.create({
    model: 'claude-sonnet-4-6',
    max_tokens: 1500,
    system: SYSTEM_PROMPT,
    messages: [{ role: 'user', content: userPrompt }],
  });

  let text = response.content
    .filter((block) => block.type === 'text')
    .map((block) => block.text)
    .join('')
    .trim();

  if (text.startsWith('```')) {
    text = text.replace(/^```(json)?/, '').replace(/```$/, '').trim();
  }

  return JSON.parse(text);
}

module.exports = async (req, res) => {
  res.setHeader('Access-Control-Allow-Origin', '*');

  if (req.method === 'OPTIONS') {
    res.setHeader('Access-Control-Allow-Methods', 'POST, OPTIONS');
    res.setHeader('Access-Control-Allow-Headers', 'Content-Type');
    res.status(204).end();
    return;
  }

  if (req.method !== 'POST') {
    res.status(405).json({ error: 'This endpoint only accepts POST requests with a JSON body: {"idea": "..."}' });
    return;
  }

  try {
    const body = typeof req.body === 'string' ? JSON.parse(req.body || '{}') : (req.body || {});
    const idea = (body.idea || '').trim();

    if (!idea) {
      res.status(400).json({ error: 'Please describe the idea you want to validate.' });
      return;
    }

    if (!process.env.ANTHROPIC_API_KEY) {
      res.status(500).json({ error: "ANTHROPIC_API_KEY is not set in this deployment's environment variables." });
      return;
    }

    const { docVecs, vectorize } = buildCorpus(posts);
    const evidence = retrieve(idea, posts, docVecs, vectorize, TOP_K);

    const result = await callClaude(idea, evidence);
    result.evidence = evidence;

    res.status(200).json(result);
  } catch (err) {
    res.status(500).json({ error: `Something went wrong: ${err.message}` });
  }
};
