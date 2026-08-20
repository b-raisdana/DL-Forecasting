# Todo Planning: Preserve Type in `options_settings`

## Context

`app/infrastructure/options_settings.py` uses a `diskcache.Cache` (`options_settings`) as a generic key-value store. Currently, values are cast back to their expected types with `cast(datetime, cached_timestamp)` rather than being validated, which means the cache is effectively untyped at runtime.

## Problems

- **No runtime type validation**: `options_settings.set(cache_key, oldest_timestamp)` accepts anything; `options_settings.get(cache_key)` returns `Any`.
- **Unsafe casts**: `cast(datetime, cached_timestamp)` silences the type checker but does not verify the value at runtime.
- **No schema for cache entries**: Keys follow an ad-hoc string convention (`oldest_timestamp_{broker}_{symbol}`) with no enforced structure or documented value type.
- **Silent corruption risk**: If a cached entry is manually deleted, corrupted, or stored with the wrong type, the code raises at runtime without a clear message.

## Goals

1. Make the value type of each cache entry explicit and enforceable.
2. Remove unsafe `cast()` calls and replace them with validated retrieval.
3. Keep the public interface of `get_oldest_available_timestamp` unchanged.

## Proposed Tasks

### Task 1: Define typed cache key constants and value contracts

- Create a small module or section that declares each cache key as a `Final` string.
- Document the expected value type for every key.

### Task 2: Add typed get/set helpers

- Introduce `get_datetime(key: str) -> datetime` and `set_datetime(key: str, value: datetime) -> None`.
- Add runtime validation (e.g., `isinstance(value, datetime)`).
- Raise a typed exception (or `TypeError`) when a cached value has the wrong type.

### Task 3: Migrate `get_oldest_available_timestamp`

- Replace `options_settings.get(cache_key)` and `cast(datetime, ...)` with `get_datetime(cache_key)`.
- Replace `options_settings.set(cache_key, oldest_timestamp)` with `set_datetime(cache_key, oldest_timestamp)`.

### Task 4: Add tests

- Test that storing and retrieving preserves the `datetime` type.
- Test that corrupted or wrong-type cached values raise the expected error.

### Task 5: Review other call sites

- Search for all uses of `options_settings` across the repo.
- Apply the typed helpers (or equivalent validation) wherever values are read or written.

## Out of Scope

- Changing the underlying cache backend.
- Changing the public signature of `get_oldest_available_timestamp`.
