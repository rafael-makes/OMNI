-- omni-core :: similarity search RPC (Step 2)
-- Apply after 0001. Idempotent (create or replace).
--
-- Returns the nearest memories to `query_embedding` by cosine distance, using
-- the HNSW index from 0001. Called from MemoryStore.retrieve() via PostgREST RPC.
--
-- Person filtering (SPEC): when `filter_person` is null, all rows are eligible;
-- when set, results are that person's records PLUS general (person is null) ones.
-- Light recency/importance boosting is applied client-side in MemoryStore.

create or replace function public.match_memories(
    query_embedding vector(768),
    match_count int default 5,
    filter_person text default null
)
returns table (
    id uuid,
    content text,
    person text,
    source text,
    location text,
    session_id text,
    importance smallint,
    created_at timestamptz,
    similarity double precision
)
language sql
stable
as $$
    select
        m.id,
        m.content,
        m.person,
        m.source,
        m.location,
        m.session_id,
        m.importance,
        m.created_at,
        1 - (m.embedding <=> query_embedding) as similarity
    from public.memories m
    where m.embedding is not null
      and (
          filter_person is null
          or m.person = filter_person
          or m.person is null
      )
    order by m.embedding <=> query_embedding
    limit match_count;
$$;
