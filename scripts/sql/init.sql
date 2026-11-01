-- Runs once when the database volume is created.
-- Tables and HNSW indexes are created by the indexer (one table per embedding model + chunk config).
CREATE EXTENSION IF NOT EXISTS vector;
