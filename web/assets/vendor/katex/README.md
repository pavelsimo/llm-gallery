# Vendored KaTeX

- Version: 0.16.22
- Source: https://github.com/KaTeX/KaTeX/releases/download/v0.16.22/katex.tar.gz
- Files kept: `katex.min.css`, `katex.min.js`, `fonts/*.woff2`
- Pruned: `.woff`/`.ttf` fonts (all supported browsers use woff2), unminified builds,
  `katex.mjs`, `contrib/` (the auto-render extension is not used — equations are
  pre-extracted at build time into `.katex-src` placeholders and rendered with
  `katex.render()` in `web/app.js`).

To upgrade: download the release tarball, copy the same files, update the version here.
