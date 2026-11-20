CREATE ROLE bookstreet_ingest LOGIN PASSWORD :'ingest_pass';
CREATE ROLE bookstreet_app    LOGIN PASSWORD :'app_pass';

GRANT CONNECT ON DATABASE vectordb TO bookstreet_ingest, bookstreet_app;
GRANT USAGE, CREATE ON SCHEMA public TO bookstreet_ingest;
GRANT USAGE ON SCHEMA public TO bookstreet_app;

ALTER DEFAULT PRIVILEGES FOR ROLE bookstreet_ingest IN SCHEMA public
  GRANT SELECT, INSERT ON TABLES TO bookstreet_app;
ALTER DEFAULT PRIVILEGES FOR ROLE bookstreet_ingest IN SCHEMA public
  GRANT USAGE, SELECT ON SEQUENCES TO bookstreet_app;

ALTER TABLE IF EXISTS documents OWNER TO bookstreet_ingest;
