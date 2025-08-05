#!/bin/bash
set -e

# Custom PostgreSQL Docker entrypoint
# Extends the official PostgreSQL entrypoint with FDW-specific setup

# Source the original entrypoint functions
source /usr/local/bin/docker-entrypoint.sh

# Custom initialization function
init_fdw_extensions() {
    echo "Initializing FDW extensions..."
    
    # Wait for PostgreSQL to be ready
    until pg_isready -U "$POSTGRES_USER" -d "$POSTGRES_DB"; do
        echo "Waiting for PostgreSQL to be ready..."
        sleep 2
    done
    
    # Create extensions
    psql -v ON_ERROR_STOP=1 --username "$POSTGRES_USER" --dbname "$POSTGRES_DB" <<-EOSQL
        -- Create required extensions
        CREATE EXTENSION IF NOT EXISTS postgres_fdw;
        CREATE EXTENSION IF NOT EXISTS file_fdw;
        CREATE EXTENSION IF NOT EXISTS pg_stat_statements;
        CREATE EXTENSION IF NOT EXISTS btree_gin;
        CREATE EXTENSION IF NOT EXISTS btree_gist;
        
        -- Try to create mysql_fdw if available
        DO \$\$
        BEGIN
            CREATE EXTENSION IF NOT EXISTS mysql_fdw;
            RAISE NOTICE 'mysql_fdw extension created successfully';
        EXCEPTION WHEN OTHERS THEN
            RAISE NOTICE 'mysql_fdw extension not available: %', SQLERRM;
        END;
        \$\$;
        
        -- Try to create multicorn if available
        DO \$\$
        BEGIN
            CREATE EXTENSION IF NOT EXISTS multicorn;
            RAISE NOTICE 'multicorn extension created successfully';
        EXCEPTION WHEN OTHERS THEN
            RAISE NOTICE 'multicorn extension not available: %', SQLERRM;
        END;
        \$\$;
        
        -- Create monitoring user
        DO \$\$
        BEGIN
            IF NOT EXISTS (SELECT FROM pg_catalog.pg_roles WHERE rolname = 'monitoring') THEN
                CREATE ROLE monitoring WITH LOGIN PASSWORD 'monitoring_password_change_me';
                GRANT CONNECT ON DATABASE ${POSTGRES_DB} TO monitoring;
                GRANT SELECT ON ALL TABLES IN SCHEMA public TO monitoring;
                GRANT SELECT ON ALL TABLES IN SCHEMA information_schema TO monitoring;
                GRANT SELECT ON ALL TABLES IN SCHEMA pg_catalog TO monitoring;
            END IF;
        END;
        \$\$;
        
        -- Create backup user
        DO \$\$
        BEGIN
            IF NOT EXISTS (SELECT FROM pg_catalog.pg_roles WHERE rolname = 'backup_user') THEN
                CREATE ROLE backup_user WITH LOGIN REPLICATION PASSWORD 'backup_password_change_me';
            END IF;
        END;
        \$\$;
        
        -- Grant necessary permissions to main user
        GRANT ALL ON SCHEMA public TO ${POSTGRES_USER};
        GRANT ALL ON ALL TABLES IN SCHEMA public TO ${POSTGRES_USER};
        GRANT ALL ON ALL SEQUENCES IN SCHEMA public TO ${POSTGRES_USER};
        GRANT ALL ON ALL FUNCTIONS IN SCHEMA public TO ${POSTGRES_USER};
        
        -- Create schemas for FDW
        CREATE SCHEMA IF NOT EXISTS fdw_external;
        CREATE SCHEMA IF NOT EXISTS fdw_staging;
        
        GRANT ALL ON SCHEMA fdw_external TO ${POSTGRES_USER};
        GRANT ALL ON SCHEMA fdw_staging TO ${POSTGRES_USER};
EOSQL
    
    echo "FDW extensions initialized successfully"
}

# Custom configuration setup
setup_custom_config() {
    echo "Setting up custom PostgreSQL configuration..."
    
    # Copy custom configuration files if they exist
    if [ -f /tmp/postgresql.conf ]; then
        echo "Copying custom postgresql.conf..."
        cp /tmp/postgresql.conf "$PGDATA/postgresql.conf"
        chown postgres:postgres "$PGDATA/postgresql.conf"
        chmod 600 "$PGDATA/postgresql.conf"
    fi
    
    if [ -f /tmp/pg_hba.conf ]; then
        echo "Copying custom pg_hba.conf..."
        cp /tmp/pg_hba.conf "$PGDATA/pg_hba.conf"
        chown postgres:postgres "$PGDATA/pg_hba.conf"
        chmod 600 "$PGDATA/pg_hba.conf"
    fi
    
    echo "Custom configuration setup completed"
}

# Health monitoring function
setup_health_monitoring() {
    echo "Setting up health monitoring..."
    
    # Create health check function
    psql -v ON_ERROR_STOP=1 --username "$POSTGRES_USER" --dbname "$POSTGRES_DB" <<-EOSQL
        CREATE OR REPLACE FUNCTION health_check()
        RETURNS TABLE(
            status text,
            connections_active integer,
            connections_max integer,
            database_size text,
            uptime interval
        ) AS \$\$
        BEGIN
            RETURN QUERY
            SELECT 
                'healthy'::text as status,
                (SELECT count(*) FROM pg_stat_activity WHERE state = 'active')::integer as connections_active,
                (SELECT setting::integer FROM pg_settings WHERE name = 'max_connections') as connections_max,
                pg_size_pretty(pg_database_size(current_database())) as database_size,
                (SELECT current_timestamp - pg_postmaster_start_time()) as uptime;
        END;
        \$\$ LANGUAGE plpgsql;
        
        -- Grant execute permission
        GRANT EXECUTE ON FUNCTION health_check() TO monitoring;
EOSQL
    
    echo "Health monitoring setup completed"
}

# Performance tuning based on available resources
tune_performance() {
    echo "Applying performance tuning..."
    
    # Get available memory (in MB)
    TOTAL_MEM=$(free -m | awk 'NR==2{print $2}')
    
    # Calculate optimal settings based on available memory
    SHARED_BUFFERS=$((TOTAL_MEM / 4))
    EFFECTIVE_CACHE_SIZE=$((TOTAL_MEM * 3 / 4))
    WORK_MEM=$((TOTAL_MEM / 100))
    MAINTENANCE_WORK_MEM=$((TOTAL_MEM / 16))
    
    # Ensure minimum values
    [ $SHARED_BUFFERS -lt 128 ] && SHARED_BUFFERS=128
    [ $EFFECTIVE_CACHE_SIZE -lt 256 ] && EFFECTIVE_CACHE_SIZE=256
    [ $WORK_MEM -lt 4 ] && WORK_MEM=4
    [ $MAINTENANCE_WORK_MEM -lt 64 ] && MAINTENANCE_WORK_MEM=64
    
    echo "Tuning PostgreSQL for ${TOTAL_MEM}MB total memory:"
    echo "  - shared_buffers: ${SHARED_BUFFERS}MB"
    echo "  - effective_cache_size: ${EFFECTIVE_CACHE_SIZE}MB"
    echo "  - work_mem: ${WORK_MEM}MB"
    echo "  - maintenance_work_mem: ${MAINTENANCE_WORK_MEM}MB"
    
    # Apply settings
    psql -v ON_ERROR_STOP=1 --username "$POSTGRES_USER" --dbname "$POSTGRES_DB" <<-EOSQL
        ALTER SYSTEM SET shared_buffers = '${SHARED_BUFFERS}MB';
        ALTER SYSTEM SET effective_cache_size = '${EFFECTIVE_CACHE_SIZE}MB';
        ALTER SYSTEM SET work_mem = '${WORK_MEM}MB';
        ALTER SYSTEM SET maintenance_work_mem = '${MAINTENANCE_WORK_MEM}MB';
        SELECT pg_reload_conf();
EOSQL
    
    echo "Performance tuning completed"
}

# Main execution
main() {
    echo "Starting custom PostgreSQL initialization..."
    
    # Check if this is the first run
    if [ "$1" = 'postgres' ] && [ -z "$POSTGRES_HOST_AUTH_METHOD" ]; then
        # Set up custom configuration before PostgreSQL starts
        if [ ! -s "$PGDATA/PG_VERSION" ]; then
            echo "First run detected, setting up custom configuration..."
            
            # Initialize database with original entrypoint
            docker_setup_env
            docker_create_db_directories
            docker_init_database_dir
            
            # Set up custom configuration
            setup_custom_config
            
            # Start PostgreSQL temporarily for setup
            docker_temp_server_start
            
            # Run custom initialization
            init_fdw_extensions
            setup_health_monitoring
            tune_performance
            
            # Stop temporary server
            docker_temp_server_stop
            
            echo "Custom initialization completed"
        fi
    fi
    
    # Continue with original entrypoint
    exec docker-entrypoint.sh "$@"
}

# Execute main function with all arguments
main "$@"