---
title: Quickstart
description: Run the command line entry point of byob.
---

First, ensure that `byob` is [installed](/byob/getting-started/installation/).

## Basic usage

After installation, you can run the application with:

```bash
byob
```

which calls `byob.main.main` and prints a greeting.

## Use the package from Python

```python
from byob.main import main

main()
```

See the [API reference](/byob/api/byob/) for every module and function.

## Profiling

byob's coarse operations are [scope-profiler](https://github.com/max-models/scope-profiler)
regions (`byob.main` here), tagged with the opt-in tag `byob`: they are
off by default and cost a fraction of a microsecond per call. A code that profiles itself with
scope-profiler turns them on with the tag:

```python
import scope_profiler as sp

sp.setup(enable_tags=["byob"])
```

or without editing the code, with `SCOPE_PROFILER_ENABLE_TAGS=byob`,
`scope-profiler run --enable-tag byob` or `enable_tags = ["byob"]` in the
TOML configuration. New regions use `@sp.profile("byob.<name>", tags=PROFILE_TAGS)`
with `PROFILE_TAGS` from the package's `_profiling` module.
