# blog.jacktcollins.com

Jekyll source for my blog, served via GitHub Pages at
[blog.jacktcollins.com](https://blog.jacktcollins.com).

## Writing a post

Add a Markdown file to `_posts/` named `YYYY-MM-DD-title.md` with front matter:

```yaml
---
layout: post
title: "Your Title"
date: 2026-05-28
author: Jack Collins
---
```

Put post images under `assets/posts/<slug>/` and reference them with an
absolute path, e.g. `![alt](/assets/posts/<slug>/image.png)`. A paragraph
immediately followed by `{:.figure-caption}` renders as a centered caption.

Math is rendered with MathJax: `$...$` for inline, `$$...$$` for display.

## Local preview (optional)

Requires Ruby + Bundler:

```bash
bundle install
bundle exec jekyll serve
```

GitHub Pages rebuilds automatically on every push to `main`.
