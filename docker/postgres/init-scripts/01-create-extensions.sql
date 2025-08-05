-- PostgreSQL FDW Extension Initialization
-- This script creates all necessary extensions and initial setup

-- Enable required extensions
CREATE EXTENSION IF NOT EXISTS postgres_fdw;
CREATE EXTENSION IF NOT EXISTS file_fdw;
CREATE EXTENSION IF NOT EXISTS pg_stat_statements;
CREATE EXTENSION IF NOT EXISTS btree_gin;
CREATE EXTENSION IF NOT EXISTS btree_gist;

-- Try to create optional extensions
DO $$
BEGIN
    -- MySQL FDW (if available)
    BEGIN
        CREATE EXTENSION IF NOT EXISTS mysql_fdw;
        RAISE NOTICE 'mysql_fdw extension created successfully';
    EXCEPTION WHEN OTHERS THEN
        RAISE NOTICE 'mysql_fdw extension not available: %', SQLERRM;
    END;
    
    -- Multicorn FDW (if available)
    BEGIN
        CREATE EXTENSION IF NOT EXISTS multicorn;
        RAISE NOTICE 'multicorn extension created successfully';
    EXCEPTION WHEN OTHERS THEN
        RAISE NOTICE 'multicorn extension not available: %', SQLERRM;
    END;
    
    -- UUID extension
    BEGIN
        CREATE EXTENSION IF NOT EXISTS "uuid-ossp";
        RAISE NOTICE 'uuid-ossp extension created successfully';
    EXCEPTION WHEN OTHERS THEN
        RAISE NOTICE 'uuid-ossp extension not available: %', SQLERRM;
    END;
    
    -- Crypto extension
    BEGIN
        CREATE EXTENSION IF NOT EXISTS pgcrypto;
        RAISE NOTICE 'pgcrypto extension created successfully';
    EXCEPTION WHEN OTHERS THEN
        RAISE NOTICE 'pgcrypto extension not available: %', SQLERRM;
    END;
END $$;

-- Create schemas for FDW organization
CREATE SCHEMA IF NOT EXISTS fdw_external;
CREATE SCHEMA IF NOT EXISTS fdw_staging;
CREATE SCHEMA IF NOT EXISTS fdw_monitoring;

-- Comment schemas
COMMENT ON SCHEMA fdw_external IS 'Schema for external data sources via FDW';
COMMENT ON SCHEMA fdw_staging IS 'Schema for staging data from external sources';
COMMENT ON SCHEMA fdw_monitoring IS 'Schema for monitoring and logging FDW operations';

-- Create monitoring tables
CREATE TABLE IF NOT EXISTS fdw_monitoring.connection_stats (
    id SERIAL PRIMARY KEY,
    server_name VARCHAR(255) NOT NULL,
    connection_time TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    status VARCHAR(50) NOT NULL,
    error_message TEXT,
    response_time_ms INTEGER,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS fdw_monitoring.query_stats (
    id SERIAL PRIMARY KEY,
    server_name VARCHAR(255) NOT NULL,
    table_name VARCHAR(255) NOT NULL,
    query_text TEXT NOT NULL,
    execution_time_ms INTEGER,
    rows_returned INTEGER,
    status VARCHAR(50) NOT NULL,
    error_message TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Create indexes for monitoring tables
CREATE INDEX IF NOT EXISTS idx_connection_stats_server_time 
    ON fdw_monitoring.connection_stats(server_name, connection_time);
    
CREATE INDEX IF NOT EXISTS idx_query_stats_server_table_time 
    ON fdw_monitoring.query_stats(server_name, table_name, created_at);

-- Create system health monitoring function
CREATE OR REPLACE FUNCTION fdw_monitoring.system_health()
RETURNS TABLE(
    metric_name TEXT,
    metric_value TEXT,
    status TEXT,
    last_updated TIMESTAMP
) AS $$
BEGIN
    RETURN QUERY
    SELECT 
        'database_size'::TEXT as metric_name,
        pg_size_pretty(pg_database_size(current_database())) as metric_value,
        CASE 
            WHEN pg_database_size(current_database()) < 1073741824 THEN 'healthy'  -- < 1GB
            WHEN pg_database_size(current_database()) < 10737418240 THEN 'warning' -- < 10GB
            ELSE 'critical'
        END as status,
        CURRENT_TIMESTAMP as last_updated
    
    UNION ALL
    
    SELECT 
        'active_connections'::TEXT,
        (SELECT count(*)::TEXT FROM pg_stat_activity WHERE state = 'active'),
        CASE 
            WHEN (SELECT count(*) FROM pg_stat_activity WHERE state = 'active') < 50 THEN 'healthy'
            WHEN (SELECT count(*) FROM pg_stat_activity WHERE state = 'active') < 100 THEN 'warning'
            ELSE 'critical'
        END,
        CURRENT_TIMESTAMP
    
    UNION ALL
    
    SELECT 
        'replication_lag_bytes'::TEXT,
        COALESCE((SELECT pg_wal_lsn_diff(pg_current_wal_lsn(), replay_lsn)::TEXT 
                 FROM pg_stat_replication LIMIT 1), '0'),
        CASE 
            WHEN COALESCE((SELECT pg_wal_lsn_diff(pg_current_wal_lsn(), replay_lsn) 
                          FROM pg_stat_replication LIMIT 1), 0) < 1048576 THEN 'healthy'  -- < 1MB
            WHEN COALESCE((SELECT pg_wal_lsn_diff(pg_current_wal_lsn(), replay_lsn) 
                          FROM pg_stat_replication LIMIT 1), 0) < 10485760 THEN 'warning' -- < 10MB
            ELSE 'critical'
        END,
        CURRENT_TIMESTAMP;
END;
$$ LANGUAGE plpgsql;

-- Create FDW connection testing function
CREATE OR REPLACE FUNCTION fdw_monitoring.test_fdw_connection(server_name TEXT)
RETURNS TABLE(
    server TEXT,
    status TEXT,
    response_time_ms INTEGER,
    error_message TEXT
) AS $$
DECLARE
    start_time TIMESTAMP;
    end_time TIMESTAMP;
    test_query TEXT;
    result RECORD;
BEGIN
    start_time := clock_timestamp();
    
    -- Construct test query based on server type
    test_query := format('SELECT 1 as test_connection FROM %I.information_schema.tables LIMIT 1', server_name);
    
    BEGIN
        EXECUTE test_query INTO result;
        end_time := clock_timestamp();
        
        RETURN QUERY SELECT 
            server_name,
            'success'::TEXT,
            EXTRACT(MILLISECONDS FROM (end_time - start_time))::INTEGER,
            NULL::TEXT;
            
        -- Log successful connection
        INSERT INTO fdw_monitoring.connection_stats (server_name, status, response_time_ms)
        VALUES (server_name, 'success', EXTRACT(MILLISECONDS FROM (end_time - start_time))::INTEGER);
        
    EXCEPTION WHEN OTHERS THEN
        end_time := clock_timestamp();
        
        RETURN QUERY SELECT 
            server_name,
            'failed'::TEXT,
            EXTRACT(MILLISECONDS FROM (end_time - start_time))::INTEGER,
            SQLERRM;
            
        -- Log failed connection
        INSERT INTO fdw_monitoring.connection_stats (server_name, status, response_time_ms, error_message)
        VALUES (server_name, 'failed', EXTRACT(MILLISECONDS FROM (end_time - start_time))::INTEGER, SQLERRM);
    END;
END;
$$ LANGUAGE plpgsql;

-- Create cleanup function for monitoring data
CREATE OR REPLACE FUNCTION fdw_monitoring.cleanup_old_stats(days_to_keep INTEGER DEFAULT 30)
RETURNS INTEGER AS $$
DECLARE
    deleted_count INTEGER;
BEGIN
    -- Clean up old connection stats
    DELETE FROM fdw_monitoring.connection_stats 
    WHERE created_at < CURRENT_TIMESTAMP - INTERVAL '1 day' * days_to_keep;
    
    GET DIAGNOSTICS deleted_count = ROW_COUNT;
    
    -- Clean up old query stats
    DELETE FROM fdw_monitoring.query_stats 
    WHERE created_at < CURRENT_TIMESTAMP - INTERVAL '1 day' * days_to_keep;
    
    GET DIAGNOSTICS deleted_count = deleted_count + ROW_COUNT;
    
    RETURN deleted_count;
END;
$$ LANGUAGE plpgsql;

-- Set up automatic cleanup (run daily)
-- Note: This would typically be set up via cron or a job scheduler
COMMENT ON FUNCTION fdw_monitoring.cleanup_old_stats(INTEGER) IS 'Cleanup function to remove old monitoring data. Run daily via scheduler.';

RAISE NOTICE 'FDW extensions and monitoring setup completed successfully';