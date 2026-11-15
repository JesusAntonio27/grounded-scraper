-- One row per (source, record, period). Append-only: the first capture of a period
-- wins and DAG retries never duplicate rows (ON CONFLICT DO NOTHING in the loader).
CREATE TABLE IF NOT EXISTS public.grounded_snapshots (
    id               bigserial PRIMARY KEY,
    source_id        text        NOT NULL,
    record_name      text        NOT NULL,
    source_url       text        NOT NULL,
    snapshot_period  date        NOT NULL CHECK (extract(day FROM snapshot_period) = 1),
    field_values     jsonb       NOT NULL DEFAULT '{}'::jsonb,  -- parsed numbers (as strings) or text
    raw_field_values jsonb       NOT NULL DEFAULT '{}'::jsonb,  -- exactly what the source showed
    field_ranges     jsonb       NOT NULL DEFAULT '{}'::jsonb,  -- {"field": [min, max]} for literal ranges
    status           text        NOT NULL CHECK (status IN ('captured', 'unavailable', 'page-changed')),
    diagnostic       text,
    captured_at      timestamptz NOT NULL,
    raw_capture      jsonb,                                     -- the full source artifact, for audit
    inserted_at      timestamptz NOT NULL DEFAULT now(),
    CONSTRAINT grounded_snapshots_grain UNIQUE (source_id, record_name, snapshot_period)
);

COMMENT ON TABLE public.grounded_snapshots IS
    'Monthly grounded-scraper snapshots. Every non-null value was backed by a literal quote from the page fetched that day.';
