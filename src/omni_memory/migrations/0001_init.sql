-- omni-core :: memories table (Step 1)
-- Apply against the omni-core Supabase Postgres (via Supabase Studio SQL editor,
-- or: psql "$OMNI_CORE_DB_URL" -f migrations/0001_init.sql).
-- Idempotent: safe to re-run.

-- pgvector: provides the `vector` type used by the embedding column.
create extension if not exists vector;

create table if not exists public.memories (
    id          uuid          primary key default gen_random_uuid(),
    content     text          not null,
    -- Nullable in Step 1 (no embeddings yet). Step 2 backfills / populates on store().
    -- Dimension 768 per SPEC; VERIFY against the chosen Gemini embedding model in Step 2
    -- (gemini-embedding-001 supports 768/1536/3072 via output_dimensionality).
    embedding   vector(768),
    person      text,                              -- null = general / household
    source      text          not null default 'conversation'
                              check (source in ('conversation', 'observation', 'system')),
    location    text,                              -- room tag if known
    session_id  text,                              -- groups memories from one conversation
    importance  smallint      not null default 3
                              check (importance between 1 and 5),
    created_at  timestamptz   not null default now()
);

-- Retrieval-support indexes.
create index if not exists memories_person_idx     on public.memories (person);
create index if not exists memories_created_at_idx on public.memories (created_at desc);
create index if not exists memories_session_id_idx on public.memories (session_id);

-- HNSW ANN index for cosine similarity (Step 2 retrieval).
-- pgvector requires a non-null vector for indexed rows; nulls are simply skipped,
-- so this is safe while Step 1 rows have null embeddings.
create index if not exists memories_embedding_hnsw_idx
    on public.memories
    using hnsw (embedding vector_cosine_ops);
