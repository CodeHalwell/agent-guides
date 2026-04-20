#!/usr/bin/env node
// Port all Jekyll-era markdown into Starlight's src/content/docs/.
//
// Strategy:
// 1. Discover source dirs (17 *_Guide folders + docs/ + root README.md).
// 2. Map each file to a kebab-case destination under src/content/docs/.
// 3. Rewrite Jekyll-style links ({{ site.baseurl }}, ../Foo_Guide/bar) to
//    Astro absolute links (/foo-guide/bar/).
// 4. Normalize frontmatter: strip Jekyll-specific keys, ensure title +
//    description, preserve or inject sensible defaults.
// 5. Strip Kramdown-only syntax ({% raw %}, {: .callout} attrs, etc.)
//    that would break MDX or render as literals.
//
// The script is idempotent: it wipes the generated subtree under
// src/content/docs/ (except index.mdx and 404.md, which are owned by the
// Astro app) before writing.

import { promises as fs } from 'node:fs';
import path from 'node:path';
import url from 'node:url';

const __filename = url.fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);
const repoRoot = path.resolve(__dirname, '..');
const contentDocs = path.join(repoRoot, 'src', 'content', 'docs');

// Folders that contain source docs (framework guides + repo-wide docs/).
const guideDirs = [
  'AG2_Guide',
  'Amazon_Bedrock_Agents_Guide',
  'Anthropic_Claude_Agent_SDK_Guide',
  'Anthropic_Claude_Agent_SDK_TypeScript_Guide',
  'AutoGen_Guide',
  'CrewAI_Guide',
  'Google_ADK_Guide',
  'Haystack_Guide',
  'LangGraph_Guide',
  'LlamaIndex_Guide',
  'Microsoft_Agent_Framework_Guide',
  'Mistral_Agents_API_Guide',
  'OpenAI_Agents_SDK_Guides',
  'OpenAI_Agents_SDK_TypeScript_Guide',
  'PydanticAI_Guide',
  'Semantic_Kernel_Guide',
  'SmolAgents_Guide',
];

// Files in repo root to port into the docs site.
const rootDocs = [
  { src: 'docs/index.md', dest: 'docs/about.md' },
  { src: 'docs/frameworks.md', dest: 'docs/frameworks.md' },
  { src: 'docs/guides.md', dest: 'docs/guides.md' },
  { src: 'docs/quick-start.md', dest: 'docs/quick-start.md' },
  { src: 'docs/update_report_april_17_2026.md', dest: 'docs/updates/april-17-2026.md' },
  { src: 'docs/update_report_april_18_2026.md', dest: 'docs/updates/april-18-2026.md' },
  { src: 'docs/update_report_april_19_2026.md', dest: 'docs/updates/april-19-2026.md' },
  { src: 'docs/update_report_april_20_2026.md', dest: 'docs/updates/april-20-2026.md' },
  { src: 'docs/update_report_april_2026.md', dest: 'docs/updates/april-2026-summary.md' },
];

/** Convert `My_Folder_Name` → `my-folder-name`. */
function toKebab(name) {
  return name.replace(/_/g, '-').toLowerCase();
}

/** Normalize a file path inside a guide. Keeps segment case for files, kebabs top folder. */
function mapPath(relPath) {
  const parts = relPath.split(path.sep);
  const [top, ...rest] = parts;
  const mappedTop = toKebab(top);
  const mappedRest = rest.map((p) => p.toLowerCase());
  return path.join(mappedTop, ...mappedRest);
}

/** Strip/Normalize frontmatter and body. Returns the final MD string. */
function normalizeContent(raw, { defaultTitle, framework, language, srcPath }) {
  let body = raw;
  let fm = {};

  // Parse existing frontmatter if present.
  const fmMatch = raw.match(/^---\n([\s\S]*?)\n---\n?/);
  if (fmMatch) {
    body = raw.slice(fmMatch[0].length);
    for (const line of fmMatch[1].split('\n')) {
      const m = line.match(/^([A-Za-z0-9_-]+):\s*(.*)$/);
      if (m) fm[m[1]] = m[2].trim();
    }
  }

  // Drop Jekyll-only keys.
  delete fm.layout;
  delete fm.permalink;
  delete fm.nav_exclude;

  // Title: prefer existing fm.title, else first `# Heading`, else default.
  if (!fm.title) {
    const h1 = body.match(/^#\s+(.+?)\s*$/m);
    fm.title = h1 ? h1[1].replace(/[\"']/g, '').trim() : defaultTitle;
  }
  // Trim enclosing quotes that Jekyll often had.
  fm.title = fm.title.replace(/^['"]|['"]$/g, '');

  // Description: first sentence/paragraph of visible prose (≤ 180 chars).
  if (!fm.description) {
    const prose = body
      .replace(/^#.+?$/gm, '')
      .replace(/```[\s\S]*?```/g, '')
      .replace(/^\s*[-*+]\s.+$/gm, '')
      .replace(/\|.*\|/g, '')
      .split(/\n\s*\n/)
      .map((p) => p.trim())
      .find((p) => p.length > 40 && !p.startsWith('!'));
    if (prose) {
      fm.description = prose
        .replace(/\[([^\]]+)\]\([^)]+\)/g, '$1')
        .replace(/\*\*|__|\*|_|`/g, '')
        .replace(/\s+/g, ' ')
        .slice(0, 180)
        .trim();
    }
  }

  // Body rewrites ----------------------------------------------------------

  // Jekyll liquid tags
  body = body.replace(/\{%\s*raw\s*%\}/g, '');
  body = body.replace(/\{%\s*endraw\s*%\}/g, '');
  body = body.replace(/\{\{\s*site\.baseurl\s*\}\}/g, '');
  body = body.replace(/\{\{\s*site\.[a-z_.]+\s*\}\}/gi, '');

  // Kramdown attribute lists: `{: .callout}` etc → drop.
  body = body.replace(/\{:\s*\.[^}]+\}/g, '');

  // Rewrite links: [..](../Foo_Guide/bar.md) → [..](/foo-guide/bar/)
  body = body.replace(
    /\]\(([^)]+?)\)/g,
    (full, href) => `](${rewriteHref(href, srcPath)})`,
  );
  // And reference-style: [...]: ../Foo_Guide/bar
  body = body.replace(
    /^(\s*\[[^\]]+\]:\s*)(\S+)(.*)$/gm,
    (full, pre, href, post) => `${pre}${rewriteHref(href, srcPath)}${post}`,
  );

  // MDX fussiness: escape stray `<` that aren't HTML. In .md this is fine,
  // but for code fences ensure they're already fenced correctly. We only
  // port to .md (not .mdx), so MD parser handles this.

  // Assemble
  const fmLines = [];
  fmLines.push(`title: ${quoteYaml(fm.title)}`);
  if (fm.description) fmLines.push(`description: ${quoteYaml(fm.description)}`);
  if (framework) fmLines.push(`framework: ${framework}`);
  if (language) fmLines.push(`language: ${language}`);
  // Preserve any other pre-existing keys the author set.
  for (const [k, v] of Object.entries(fm)) {
    if (['title', 'description'].includes(k)) continue;
    fmLines.push(`${k}: ${v}`);
  }
  return `---\n${fmLines.join('\n')}\n---\n\n${body.trimStart()}\n`;
}

function quoteYaml(s) {
  // Always double-quote and escape inner double quotes; trim to ~200 chars.
  const safe = s.replace(/\\/g, '\\\\').replace(/"/g, '\\"').replace(/\n/g, ' ');
  return `"${safe}"`;
}

function rewriteHref(href, srcPath) {
  if (!href) return href;
  const raw = href.trim();

  // External / anchor / mailto — leave alone.
  if (/^(https?:|mailto:|#|tel:)/i.test(raw)) return href;
  // Image paths or other assets — leave alone for now.
  if (/\.(png|jpe?g|gif|svg|webp|pdf|zip|ico)(\?|#|$)/i.test(raw)) return href;

  // Jekyll baseurl absolute → root-relative.
  let h = raw.replace(/^\/AgentGuides\//, '/');

  // Split off any fragment/query early — work on path, reattach at end.
  const hashIdx = h.search(/[#?]/);
  const pathOnly = hashIdx >= 0 ? h.slice(0, hashIdx) : h;
  const frag = hashIdx >= 0 ? h.slice(hashIdx) : '';

  // Absolute path targeting a Foo_Guide folder → kebab-case + dir URL.
  // Matches both absolute ("/OpenAI_Agents_SDK_Guides/...") and bare
  // ("OpenAI_Agents_SDK_Guides/..." or "../OpenAI_Agents_SDK_Guides/...").
  const guidePattern = /(?:^|\/)((?:[A-Z][A-Za-z0-9]*_)+(?:Guide|Guides))(\/|$)/;
  if (guidePattern.test(pathOnly)) {
    const cleaned = pathOnly.replace(/^\.\.\//, '').replace(/^\//, '');
    const segs = cleaned.split('/').filter(Boolean);
    if (segs.length > 0) {
      segs[0] = toKebab(segs[0]);
      for (let i = 1; i < segs.length; i++) segs[i] = segs[i].toLowerCase();
    }
    let last = segs[segs.length - 1] ?? '';
    last = last.replace(/\.mdx?$/i, '');
    if (last.toLowerCase() === 'readme' || last === '') {
      segs.pop();
    } else {
      segs[segs.length - 1] = last;
    }
    return '/' + segs.join('/') + (segs.length ? '/' : '') + frag;
  }

  // Any relative path ending in .md — strip .md, add trailing slash.
  // Handles: foo.md, ./foo.md, ../foo.md, ./sub/foo.md, ../python/foo.md, README.md.
  if (/\.mdx?$/i.test(pathOnly)) {
    // Preserve leading ./ or ../ sequences.
    const leadingMatch = pathOnly.match(/^(?:\.\.?\/)+/);
    const lead = leadingMatch ? leadingMatch[0] : './';
    const rest = pathOnly.slice(lead.length);
    const segs = rest.split('/').filter(Boolean).map((s, i, arr) =>
      i === arr.length - 1 ? s.replace(/\.mdx?$/i, '').toLowerCase() : s.toLowerCase(),
    );
    let tail = segs.join('/');
    const lastSeg = segs[segs.length - 1] ?? '';
    if (lastSeg.toLowerCase() === 'readme') {
      // README.md → index of that folder
      segs.pop();
      tail = segs.join('/');
      return (lead + (tail ? tail + '/' : '')).replace(/\/+/g, '/').replace(/^\/?/, (m) => m) + frag;
    }
    return (lead + tail + '/') + frag;
  }

  return href;
}

async function walk(dir) {
  const out = [];
  async function rec(d) {
    let entries;
    try {
      entries = await fs.readdir(d, { withFileTypes: true });
    } catch {
      return;
    }
    for (const e of entries) {
      const full = path.join(d, e.name);
      if (e.isDirectory()) {
        if (e.name.startsWith('.') || e.name === 'node_modules') continue;
        await rec(full);
      } else if (e.isFile() && /\.mdx?$/i.test(e.name)) {
        out.push(full);
      }
    }
  }
  await rec(dir);
  return out;
}

async function portGuideDir(guideDir) {
  const srcBase = path.join(repoRoot, guideDir);
  const framework = toKebab(guideDir.replace(/_Guides?$/, ''));
  const files = await walk(srcBase);

  let wrote = 0;
  for (const file of files) {
    const rel = path.relative(repoRoot, file);
    const mapped = mapPath(rel);
    const dest = path.join(contentDocs, mapped);

    // Language detection from the second folder segment.
    const segs = rel.split(path.sep);
    const langSeg = segs[1]?.toLowerCase();
    const language = ['python', 'typescript', 'dotnet', 'go'].includes(langSeg)
      ? langSeg
      : undefined;

    const raw = await fs.readFile(file, 'utf8');
    const titleFallback = path
      .basename(file, path.extname(file))
      .replace(/[_-]+/g, ' ')
      .replace(/\b\w/g, (c) => c.toUpperCase());

    // README.md at a guide/language root becomes index.md (framework landing).
    let finalDest = dest;
    if (/(^|[\\/])readme\.mdx?$/i.test(mapped)) {
      finalDest = path.join(path.dirname(dest), 'index.md');
    }

    const normalized = normalizeContent(raw, {
      defaultTitle: titleFallback,
      framework,
      language,
      srcPath: rel,
    });

    await fs.mkdir(path.dirname(finalDest), { recursive: true });
    await fs.writeFile(finalDest, normalized);
    wrote++;
  }
  return wrote;
}

async function portRootDocs() {
  let wrote = 0;
  for (const { src, dest } of rootDocs) {
    const abs = path.join(repoRoot, src);
    try {
      const raw = await fs.readFile(abs, 'utf8');
      const finalDest = path.join(contentDocs, dest);
      const titleFallback = path
        .basename(dest, path.extname(dest))
        .replace(/[_-]+/g, ' ')
        .replace(/\b\w/g, (c) => c.toUpperCase());
      const normalized = normalizeContent(raw, {
        defaultTitle: titleFallback,
        srcPath: src,
      });
      await fs.mkdir(path.dirname(finalDest), { recursive: true });
      await fs.writeFile(finalDest, normalized);
      wrote++;
    } catch (e) {
      console.warn(`  (skip) ${src}: ${e.message}`);
    }
  }
  return wrote;
}

async function cleanGeneratedSubtree() {
  // Keep: index.mdx (authored), 404.md (authored). Remove everything else
  // so the port is deterministic on every run.
  const entries = await fs.readdir(contentDocs, { withFileTypes: true });
  for (const e of entries) {
    if (e.name === 'index.mdx' || e.name === '404.md') continue;
    const p = path.join(contentDocs, e.name);
    await fs.rm(p, { recursive: true, force: true });
  }
}

async function main() {
  console.log('→ Cleaning previously-ported content…');
  await cleanGeneratedSubtree();
  let total = 0;
  for (const d of guideDirs) {
    const n = await portGuideDir(d);
    console.log(`  ${d.padEnd(46)} ${n.toString().padStart(3)} files`);
    total += n;
  }
  const rootN = await portRootDocs();
  console.log(`  (root docs)                                     ${rootN.toString().padStart(3)} files`);
  total += rootN;
  console.log(`→ Ported ${total} files into src/content/docs/.`);
}

main().catch((e) => {
  console.error(e);
  process.exit(1);
});
