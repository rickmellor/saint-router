# goorouter — OpenSpec

This project uses [OpenSpec](https://github.com/Fission-AI/OpenSpec) for spec-driven development.

## Layout

- `specs/` — current behavior of the system, organized by capability. Empty until the first change is archived.
- `changes/` — proposed changes. Each change folder contains:
  - `proposal.md` — intent + scope (the why and what)
  - `design.md` — technical decisions and architecture
  - `tasks.md` — numbered implementation checklist
  - `specs/` — delta requirements showing what `openspec/specs/` will look like after the change is accepted

## Active changes

- [`add-initial-router/`](changes/add-initial-router/) — v1 spec for the router itself.
- [`add-cost-tracking/`](changes/add-cost-tracking/) — deferred; placeholder for the eventual cost-tracking feature.

## Change lifecycle

1. **Draft** — proposal + design written, scenarios drafted under `changes/<name>/specs/`.
2. **Approved** — owner sign-off on the proposal/design.
3. **Implementing** — `tasks.md` filled in; code changes land per the checklist.
4. **Archived** — proposed deltas merged into `openspec/specs/`; the change folder is removed (or moved to an archive subfolder, per project preference).
