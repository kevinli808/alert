-- Users table for the Alert backend (Databricks SQL / Unity Catalog).
--
-- Run once in Databricks: SQL Editor -> pick your SQL warehouse -> paste -> Run.
-- The backend also runs the CREATE TABLE IF NOT EXISTS below on its first login, so only the
-- schema strictly has to exist. Change `workspace.alert` if you set a different DATABRICKS_USERS_TABLE.

CREATE SCHEMA IF NOT EXISTS workspace.alert
  COMMENT 'Alert app data';

CREATE TABLE IF NOT EXISTS workspace.alert.users (
  id            STRING    NOT NULL COMMENT 'Auth0 user id (the access token sub claim)',
  email         STRING,
  username      STRING,
  created_at    TIMESTAMP NOT NULL,
  last_login_at TIMESTAMP NOT NULL
) COMMENT 'Alert app users, one row per Auth0 account';

-- Check who has logged in:
-- SELECT * FROM workspace.alert.users ORDER BY last_login_at DESC;
