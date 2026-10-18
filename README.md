# shauryasen.com

Source for my personal site and blog: Hugo + PaperMod, with a custom home page.

- Home page: `layouts/index.html` + `assets/css/extended/home.css`. Bio in `content/_index.md`, project cards in `data/projects.yaml`, media in `static/img/projects/`.
- Posts: Markdown files in `content/posts/` (written in Obsidian). Pushing to `main` rebuilds and deploys through `.github/workflows/`.
- Resume: `static/resume.pdf` is built from `shaurya-hub/career/resume/sim/` with the phone number removed. Rebuild it with `tools/build_web_resume.sh` after every resume change.

## Writing a new post

Create a `.md` file in `content/posts/` with this front matter:

```yaml
---
title: "Your Post Title"
date: 2026-03-13
tags: ["tag1", "tag2"]
summary: "A brief summary of the post."
draft: true
---
```

This repo is public, so `draft: true` hides a post from the site but not from GitHub. Keep unfinished posts in `~/Dev/blog-drafts/` and move them here when they are ready. Preview locally with `hugo server -D`.
