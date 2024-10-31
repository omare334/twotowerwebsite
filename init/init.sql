-- init.sql

CREATE TABLE IF NOT EXISTS results (
    query TEXT,
    passage_text TEXT,
    negative_sample TEXT
);

COPY results(query, passage_text, negative_sample) 
FROM '/docker-entrypoint-initdb.d/results_negative.csv' 
DELIMITER ',' 
CSV HEADER;