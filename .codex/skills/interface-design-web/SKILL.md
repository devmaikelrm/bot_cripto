---
name: interface-design-web
description: Design, refactor, and review production web interfaces (HTML/CSS/JS/TS/React/Tailwind) in local repos or GitHub-cloned projects. Use when users ask to create UI screens, improve UX/accessibility, modernize visual design, or translate mockups/ideas into implementation-ready frontend code.
---

# Interface Design Web

## Workflow

1. Identify stack and constraints from repo context.
2. Define visual direction before coding:
- page goal (conversion, dashboard, branding, internal tool)
- primary audience
- device priority (mobile-first or desktop-first)
- accessibility level (minimum WCAG AA)
3. Build/update UI with clear structure:
- typography scale
- spacing system
- color tokens
- component states (default, hover, focus, disabled, loading)
4. Validate before finishing:
- responsive layout (mobile/tablet/desktop)
- keyboard focus visibility
- color contrast
- empty/error/loading states
- no broken interactions

## Output Standard

- Keep changes in real project files, no pseudocode.
- Prefer reusable components over duplicated markup.
- Add CSS variables/tokens when introducing a new visual system.
- Preserve existing design system conventions when repo already has one.
- If no design system exists, create one minimally (colors, spacing, typography, radii, shadows).

## GitHub Repo Mode

When work starts from a GitHub project, use `references/github-ui-checklist.md` to:
- audit existing UI quickly
- estimate impact/risk before refactor
- apply safe rollout strategy (small PR-sized increments)

## Quality Gates

- Build passes (`npm run build`/equivalent if available).
- Lint passes (`npm run lint`/equivalent if available).
- Main screen renders without console errors.
- Interactive elements are reachable by keyboard.
