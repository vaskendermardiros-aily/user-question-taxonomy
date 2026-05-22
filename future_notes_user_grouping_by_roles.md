# User Grouping by Roles — Options for BERTopic Cohort Selection

Context: ~5000 unique `user_roles` across all users, each stored as a semicolon-separated string.
Roles follow a hierarchical format: `{module}.{qualifier}` (e.g. `rnd.clinops`, `fin.no_execution`, `gra.pathways`).
Generic infrastructure roles (`tenant.*`, `*.global`, `app.access`) are shared by almost everyone and should be stripped before any similarity computation.

---

## Option 1 — Role prefix extraction (simplest)

Strip everything after the first `.` to get the module set per user: `fin`, `rnd`, `gra`, `ppl`, `supply`, etc.
Users sharing the same set of module prefixes are functionally equivalent.

Use as a **pre-filter**: e.g. "all users who have at least one `rnd.*` role" → run BERTopic on that cohort.
No matrix math needed.

**Best for:** quick cohort selection, readable groupings, low implementation cost.

---

## Option 2 — Jaccard similarity on sparse role vectors

Represent each user as a binary row vector over the full role universe (N users × 5000 roles, sparse).
Compute pairwise Jaccard similarity:

```
J(A, B) = |A ∩ B| / |A ∪ B|
```

Threshold (e.g. Jaccard > 0.7) defines "same team".

Because each user has only ~20–30 roles, 99.5% of the matrix is zeros — sparse operations make this tractable even for tens of thousands of users.

**Important:** strip generic infrastructure roles first (`tenant.*`, `*.global`, `app.access`) — they inflate the intersection artificially and make unrelated users look similar.

**Best for:** data-driven team detection without relying on any role naming convention.

---

## Option 3 — Skip user grouping for BERTopic entirely

Fit BERTopic globally on all users. Topics emerge from the content.
Use roles **post-hoc** as an analytical lens: "users with `rnd.*` roles all cluster around topic X."

Grouping becomes an interpretation tool on the output, not a prerequisite to running the model.

**Best for:** exploratory runs, validating topic quality before investing in cohort selection.

---

## Recommendation

Combine Option 1 and Option 3:
1. Use role prefix as a quick filter to select a same-domain cohort (e.g. all `rnd.*` users).
2. Fit BERTopic globally on that cohort.
3. Use roles post-hoc to validate and interpret topic clusters.

Only move to Option 2 if you need precise team boundaries across a large, heterogeneous user base.
