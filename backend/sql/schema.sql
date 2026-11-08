-- =====================================================================================
-- Alert — Supabase PostgreSQL schema
-- Run in the Supabase SQL editor (Dashboard → SQL Editor → New query). Idempotent.
--
-- Access model
--   * The FastAPI backend connects with the service-role/secret key (server-side only). That key
--     bypasses RLS, so FastAPI enforces "users only see their own rows" by always filtering on the
--     Auth0 user id taken from a verified JWT.
--   * RLS is still ENABLED on every table as defense in depth: the anon key can read/write nothing,
--     and if you later enable Supabase "Third-party auth → Auth0", a browser holding an Auth0 token
--     can only touch rows whose user_id equals the token's `sub` claim.
-- =====================================================================================

create extension if not exists pgcrypto;  -- gen_random_uuid()

-- -------------------------------------------------------------------------------------
-- profiles: one row per Auth0 user. No passwords are ever stored here (Auth0 owns credentials).
-- -------------------------------------------------------------------------------------
create table if not exists public.profiles (
    id          text primary key check (char_length(id) between 1 and 255),  -- Auth0 `sub`
    email       text check (email is null or char_length(email) <= 320),
    username    text check (username is null or char_length(username) <= 100),
    created_at  timestamptz not null default now()
);

-- -------------------------------------------------------------------------------------
-- user_settings: one row per user holding the frontend's settings document (sound rules,
-- custom visual states, state priority order, UI mode). The backend validates the shape
-- before writing; the database only enforces "an object, of a sane size".
--
-- This replaces the earlier per-sound `alert_preferences` table. If you already ran an older
-- version of this script, you can remove it by hand:  drop table if exists public.alert_preferences;
-- -------------------------------------------------------------------------------------
create table if not exists public.user_settings (
    user_id     text primary key references public.profiles (id) on delete cascade,
    settings    jsonb not null check (jsonb_typeof(settings) = 'object' and octet_length(settings::text) <= 200000),
    updated_at  timestamptz not null default now()
);

-- -------------------------------------------------------------------------------------
-- alert_history: one row per detected sound EVENT (not per audio chunk). No audio is stored.
-- -------------------------------------------------------------------------------------
create table if not exists public.alert_history (
    id           uuid primary key default gen_random_uuid(),
    user_id      text not null references public.profiles (id) on delete cascade,
    sound_type   text not null check (sound_type ~ '^[a-z][a-z0-9_]{1,49}$'),
    confidence   double precision not null check (confidence >= 0 and confidence <= 1),
    detected_at  timestamptz not null default now()
);
create index if not exists alert_history_user_detected_idx
    on public.alert_history (user_id, detected_at desc);

-- -------------------------------------------------------------------------------------
-- Keep user_settings.updated_at current on every UPDATE.
-- -------------------------------------------------------------------------------------
create or replace function public.set_updated_at()
returns trigger
language plpgsql
set search_path = ''
as $$
begin
    new.updated_at := now();
    return new;
end;
$$;

drop trigger if exists user_settings_set_updated_at on public.user_settings;
create trigger user_settings_set_updated_at
    before update on public.user_settings
    for each row execute function public.set_updated_at();

-- -------------------------------------------------------------------------------------
-- Row Level Security
-- -------------------------------------------------------------------------------------
alter table public.profiles          enable row level security;
alter table public.user_settings     enable row level security;
alter table public.alert_history     enable row level security;

-- The anon role gets nothing at all.
revoke all on public.profiles, public.user_settings, public.alert_history from anon;

-- Policies for the `authenticated` role. These only matter if Supabase third-party auth with Auth0
-- is configured (Auth0 tokens must carry "role": "authenticated" — see README). `sub` is the Auth0
-- user id. `(select auth.jwt())` is evaluated once per statement instead of once per row.

drop policy if exists "profiles: owner can read"   on public.profiles;
drop policy if exists "profiles: owner can insert" on public.profiles;
drop policy if exists "profiles: owner can update" on public.profiles;
create policy "profiles: owner can read"   on public.profiles for select to authenticated
    using (id = (select auth.jwt() ->> 'sub'));
create policy "profiles: owner can insert" on public.profiles for insert to authenticated
    with check (id = (select auth.jwt() ->> 'sub'));
create policy "profiles: owner can update" on public.profiles for update to authenticated
    using (id = (select auth.jwt() ->> 'sub'))
    with check (id = (select auth.jwt() ->> 'sub'));

drop policy if exists "user_settings: owner full access" on public.user_settings;
create policy "user_settings: owner full access" on public.user_settings for all to authenticated
    using (user_id = (select auth.jwt() ->> 'sub'))
    with check (user_id = (select auth.jwt() ->> 'sub'));

drop policy if exists "alert_history: owner can read"   on public.alert_history;
drop policy if exists "alert_history: owner can delete" on public.alert_history;
create policy "alert_history: owner can read"   on public.alert_history for select to authenticated
    using (user_id = (select auth.jwt() ->> 'sub'));
create policy "alert_history: owner can delete" on public.alert_history for delete to authenticated
    using (user_id = (select auth.jwt() ->> 'sub'));
-- History rows are written only by the backend (service role), never directly by clients.
