# Repository agent guidance

- Follow `COMMIT_CONVENTION.md` for every commit in this repository.
- A commit or push requires explicit user authorization; do not infer it from
  ordinary implementation work.
- Keep Git limited to production code, compact documentation, schemas,
  manifests, and dependency/setup files. Raw images, model weights, generated
  outputs, frozen reviews, and historical runners belong only in the ignored
  Drive bundle.
- Never overwrite frozen review evidence. New computation must use a new empty
  output directory and preserve input/output hashes.
