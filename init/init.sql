-- init.sql

-- Create role if it doesn't exist
DO
$$
BEGIN
    IF NOT EXISTS (
        SELECT FROM pg_catalog.pg_roles
        WHERE rolname = 'omareweis') THEN
        CREATE ROLE omareweis WITH LOGIN PASSWORD 'ommaha260801';
    END IF;
END
$$ LANGUAGE plpgsql;

-- Create the database if it doesn't exist
DO
$$
BEGIN
    IF NOT EXISTS (
        SELECT FROM pg_database
        WHERE datname = 'passages') THEN
        CREATE DATABASE passages OWNER omareweis;
    END IF;
END
$$ LANGUAGE plpgsql;

-- Connect to the database (handled automatically by docker-compose with POSTGRES_DB)
-- Create the table if it doesn't exist
CREATE TABLE IF NOT EXISTS results (
    query TEXT,
    passage_text TEXT,
    negative_sample TEXT
);

-- Grant all privileges on the results table to the user omareweis
GRANT ALL PRIVILEGES ON TABLE results TO omareweis;
