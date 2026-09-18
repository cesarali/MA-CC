---
name: upload-results
description: Upload existing MA-CC result files or directories to the cloud bucket behind results/aggregation_results and verify the uploaded data. Use for requests to send, copy, or back up results to this bucket; does not run experiments or analysis.
---

# Upload results

Support requests such as “upload this ZIP to the cloud” or “put this analysis
folder in aggregation_results.” A request to upload specified data authorizes
that upload; merely asking how uploads work does not authorize a transfer.

## Confirm the environment's connection

On the local machine, `results/aggregation_results` is a read-only rclone
mount of `r2:agent-swarm-control-research/aggregation_results`. Upload directly
with rclone, rather than copying into the mount. Its directory cache may take
about a minute to show newly uploaded objects.

Use the existing rclone installation and configuration. Do not print
credentials, change the mount to writable, or configure a new remote for an
ordinary upload. Never print or commit `.env` contents, rclone configuration
contents, access keys, tokens, or credential-bearing command arguments.
Use credentials only through the environment's existing configuration.

Before a transfer, confirm the current environment has a known connection:
inspect `findmnt -T results/aggregation_results -o TARGET,SOURCE,FSTYPE`
and `rclone listremotes` (remote names only). Check that the mount target is
the requested folder, rather than a parent filesystem. A mount source
identifies the bucket mapping; a remote name alone does not establish which
bucket should receive the data. An explicit existing environment-specific
mapping supplied by the user or repository instructions can also establish
the destination when no mount is present.

The mapping above is known on this local machine. Do not assume it applies
on Potsdam or another machine just because the repository was cloned there.
Use the commands below only after confirming this mapping. If no known
connection is available, report that cloud upload is not configured here and
ask for the intended connection without requesting credentials in chat.
Do not create a remote, copy credentials between environments, or start
automatic background synchronization. Upload only the requested data.

## Transfer

- Confirm the source exists and retain the original.
- Preserve its basename under the bucket's `aggregation_results/` prefix
  unless the user specifies another destination. Upload a ZIP as a ZIP;
  extract only if requested. Retain the top-level name for a directory.
- Use `--immutable` to avoid replacing different existing data. If a
  collision fails, resolve a new name or obtain explicit overwrite
  authorization. Do not use `sync`, delete objects, or retry an immutable
  collision without protection.
- Follow the execution environment's network approval rules. For a
  sandbox-related network failure, request escalation for the same scoped
  operation. Keep retries bounded and report unresolved failures.

Single-file example, from the repository root:

```bash
rclone copyto results/blackboard-checkpoint-small-test-01_analysis.zip \
  r2:agent-swarm-control-research/aggregation_results/blackboard-checkpoint-small-test-01_analysis.zip \
  --immutable --contimeout 10s --timeout 30s --retries 1 --low-level-retries 1
```

Directory example:

```bash
rclone copy results/my-analysis \
  r2:agent-swarm-control-research/aggregation_results/my-analysis \
  --immutable --contimeout 10s --timeout 30s --retries 1 --low-level-retries 1
```

Quote actual source and destination arguments when paths contain spaces.
For long transfers, add `--stats 10s --stats-one-line` to show progress.

## Verify and report

For a file, compare the local SHA-256 checksum with the checksum computed
from downloading the remote object:

```bash
sha256sum results/blackboard-checkpoint-small-test-01_analysis.zip
rclone hashsum SHA-256 \
  r2:agent-swarm-control-research/aggregation_results/blackboard-checkpoint-small-test-01_analysis.zip \
  --download --contimeout 10s --timeout 30s --retries 1 --low-level-retries 1
```

For a directory, use `rclone check SOURCE REMOTE_DIRECTORY --download
--one-way` with the same timeout and retry options. This checks source files
against their remote copies while permitting extra remote files.

Report the cloud destination, whether verification passed, and that the
original remains in place. If verification fails or cannot finish, distinguish
the completed transfer from the unverified data. Do not use a cached mount
listing as proof of successful upload.
