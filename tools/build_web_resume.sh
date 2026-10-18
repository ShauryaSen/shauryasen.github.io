#!/usr/bin/env bash
# Build the resume the site serves (static/resume.pdf) from the sim resume, with the phone number removed.
# Run after every resume change: tools/build_web_resume.sh
set -euo pipefail

src="$HOME/Dev/shaurya-hub/career/resume/sim/Shaurya_Sen_Resume_Sim.tex"
site="$(cd "$(dirname "$0")/.." && pwd)"
work="$(mktemp -d)"
trap 'rm -rf "$work"' EXIT

# Drop "<phone> $|$" from the header; keep \small so the rest of the line is unchanged.
sed -E 's/\\small [0-9]{3}-[0-9]{3}-[0-9]{4} \$\|\$/\\small/' "$src" > "$work/Shaurya_Sen_Resume.tex"
if grep -qE '[0-9]{3}-[0-9]{3}-[0-9]{4}' <(grep -v '^%' "$work/Shaurya_Sen_Resume.tex"); then
  echo "phone number still present; header format changed?" >&2
  exit 1
fi

(cd "$work" && latexmk -pdf -interaction=nonstopmode -quiet Shaurya_Sen_Resume.tex >/dev/null)
cp "$work/Shaurya_Sen_Resume.pdf" "$site/static/resume.pdf"
echo "wrote $site/static/resume.pdf"
