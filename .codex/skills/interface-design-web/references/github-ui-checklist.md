# GitHub UI Checklist

## 1) Discovery

- Confirm framework and build tool (`package.json`, lockfile, config files).
- Locate UI entry points (`src/pages`, `src/app`, `src/components`, `public`).
- Find style system (`tailwind.config`, css vars, theme provider, design tokens).

## 2) Baseline Audit

- Record current UX issues:
  - low contrast
  - inconsistent spacing
  - broken responsive blocks
  - missing focus styles
  - overloaded screens
- Capture before/after scope as small increments.

## 3) Implementation Rules

- Do not rewrite entire app in one shot.
- Start with the target screen and shared primitives first.
- Keep semantic HTML and proper ARIA only where needed.
- Avoid decorative complexity that hurts readability/performance.

## 4) Verification

- Run build/lint/tests available in repo.
- Validate main breakpoints: 360px, 768px, 1024px, 1440px.
- Validate keyboard navigation and focus order.
- Confirm no regressions in unchanged screens.

## 5) Delivery

- Summarize changed files and reasoning.
- Document new tokens/components introduced.
- List optional next improvements as separate follow-up tasks.
