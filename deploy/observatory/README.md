# Hosted Blackboard Observatory - image

`build-image.sh <registry/repo> <tag> [--insecure]` builds the image without a container daemon, from a
**pristine `git archive` of HEAD** (override with `MA_CC_BUILD_REF`) rather than the working tree: the package
(installed `--no-deps`) and the pinned runtime set in `requirements.txt` go into one reproducible layer under
`/opt/observatory/site-packages`, which `crane append` puts on a digest-pinned `python:3.12-slim-bookworm`.
`crane mutate` then sets the entrypoint (`python3 -m mas_cc.blackboard_dashboard`), uid 1000, and the environment
(`PYTHONPATH`, `MPLBACKEND=Agg`, `MA_CC_DASHBOARD_CACHE=/cache`). The build host needs python 3.12.

The runtime set is deliberately not the package's declared dependencies: those include torch and
sentence-transformers, which the dashboard never imports (174 MB image instead of several GB).

Run arguments are the dashboard's own, e.g.

    --host 0.0.0.0 --allowed-host observatory.example --catalog r2://bucket/prefix --catalog /studies --access-log

Object-store access comes from the environment: `AWS_ACCESS_KEY_ID`, `AWS_SECRET_ACCESS_KEY`,
`MA_CC_R2_ENDPOINT` (read-only credentials are enough; nothing is ever written). Mount a writable directory at
`/cache` and at `/tmp`; the root filesystem can be read-only.

First build: `git-722343d5cfdc` -> `sha256:410776280f96abbd9b01cabd0fa7aa91fa2c95cb127cc4711b2b2587dd65cd6a`,
verified by content (a sentence only this build contains; the three new modules present; torch absent) and signed.

## Verify what you built, by digest

An image that reports success is not an image that contains your change. This build shipped stale assets once:
a leftover `build/lib` from an earlier `pip install` keeps its own copy of every data file, setuptools re-copies
only when the source is *newer*, and a stale copy with a newer mtime was packaged in silence. Building from a
git export removes that path, but check anyway - pick a sentence only this build contains:

    crane export <repo>@<digest> - | grep -a -c '<a sentence only this build contains>'   # expect 1
    crane export <repo>@<previous digest> - | grep -a -c '<the same sentence>'            # expect 0
    crane export <repo>@<digest> - | grep -a -c '<a sentence both builds contain>'        # expect 1

Use a sentence from the SOURCE, not from the commit message: a commit message is not in the image, and grepping
for one returns 0 for a perfectly good build.
