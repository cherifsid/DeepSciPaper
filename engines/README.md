# Bundled Research Backend

`gpt-researcher/` is a regular source directory committed inside DeepSciPaper,
not a Git submodule. It includes the local research-engine modifications used
by the app. A normal clone contains the complete backend; no separate fork is
required to install or run DeepSciPaper.

Upstream: https://github.com/assafelovic/gpt-researcher

Imported local revision: `5983faf` (including DeepSciPaper-specific changes).
The upstream license and attribution remain in `gpt-researcher/LICENSE` and
the bundled source files. Future backend edits are committed and pushed with
DeepSciPaper, just like any other application code.

The previous nested Git metadata is retained locally in the ignored
`.git-backups/` directory during conversion. It is not needed at runtime.
