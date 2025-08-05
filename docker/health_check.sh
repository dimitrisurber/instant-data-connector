#!/bin/bash
# Health check script for the Data Connector application
set -e

# Configuration
HEALTH_ENDPOINT="${HEALTH_ENDPOINT:-http://localhost:8000/health}"
TIMEOUT="${TIMEOUT:-10}"
MAX_RETRIES="${MAX_RETRIES:-3}"
RETRY_DELAY="${RETRY_DELAY:-2}"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Logging function
log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $1"
}

# Health check function
check_health() {
    local endpoint="$1"
    local timeout="$2"
    
    log "Checking health endpoint: $endpoint"
    
    # Use curl to check the health endpoint
    if command -v curl >/dev/null 2>&1; then
        response=$(curl -s -f -m "$timeout" "$endpoint" 2>/dev/null)
        return $?
    elif command -v wget >/dev/null 2>&1; then
        response=$(wget -qO- --timeout="$timeout" "$endpoint" 2>/dev/null)
        return $?
    else
        log "ERROR: Neither curl nor wget is available"
        return 1
    fi
}

# Parse health response
parse_health_response() {
    local response="$1"
    
    if command -v jq >/dev/null 2>&1; then
        # Parse JSON response with jq
        status=$(echo "$response" | jq -r '.status // "unknown"')
        timestamp=$(echo "$response" | jq -r '.timestamp // "unknown"')
        version=$(echo "$response" | jq -r '.version // "unknown"')
        
        log "Status: $status"
        log "Timestamp: $timestamp"
        log "Version: $version"
        
        # Check if status is healthy
        if [ "$status" = "healthy" ] || [ "$status" = "ok" ]; then
            return 0
        else
            return 1
        fi
    else
        # Simple string check if jq is not available
        if echo "$response" | grep -q "healthy\|ok"; then
            log "Basic health check passed"
            return 0
        else
            log "Basic health check failed"
            return 1
        fi
    fi
}

# Database connectivity check
check_database() {
    log "Checking database connectivity..."
    
    # Check if we can connect to PostgreSQL
    if command -v pg_isready >/dev/null 2>&1; then
        local db_host="${DATABASE_HOST:-localhost}"
        local db_port="${DATABASE_PORT:-5432}"
        local db_user="${DATABASE_USER:-connector_user}"
        local db_name="${DATABASE_NAME:-instant_connector}"
        
        if pg_isready -h "$db_host" -p "$db_port" -U "$db_user" -d "$db_name" >/dev/null 2>&1; then
            log "Database connectivity: OK"
            return 0
        else
            log "Database connectivity: FAILED"
            return 1
        fi
    else
        log "pg_isready not available, skipping database check"
        # Don't fail the health check if pg_isready is not available
        return 0
    fi
}

# Redis connectivity check
check_redis() {
    log "Checking Redis connectivity..."
    
    if command -v redis-cli >/dev/null 2>&1; then
        local redis_host="${REDIS_HOST:-localhost}"
        local redis_port="${REDIS_PORT:-6379}"
        local redis_password="${REDIS_PASSWORD:-}"
        
        if [ -n "$redis_password" ]; then
            if redis-cli -h "$redis_host" -p "$redis_port" -a "$redis_password" --no-auth-warning ping >/dev/null 2>&1; then
                log "Redis connectivity: OK"
                return 0
            else
                log "Redis connectivity: FAILED"
                return 1
            fi
        else
            if redis-cli -h "$redis_host" -p "$redis_port" ping >/dev/null 2>&1; then
                log "Redis connectivity: OK"
                return 0
            else
                log "Redis connectivity: FAILED"
                return 1
            fi
        fi
    else
        log "redis-cli not available, skipping Redis check"
        # Don't fail the health check if redis-cli is not available
        return 0
    fi
}

# Application-specific health checks
check_application_health() {
    log "Performing application-specific health checks..."
    
    # Check if Python process is running
    if pgrep -f "uvicorn.*instant_connector" >/dev/null; then
        log "Application process: RUNNING"
    else
        log "Application process: NOT FOUND"
        return 1
    fi
    
    # Check if log directory exists and is writable
    local log_dir="${LOG_DIR:-/app/logs}"
    if [ -d "$log_dir" ] && [ -w "$log_dir" ]; then
        log "Log directory: OK"
    else
        log "Log directory: NOT ACCESSIBLE"
        return 1
    fi
    
    # Check if data directory exists and is writable
    local data_dir="${DATA_DIR:-/app/data}"
    if [ -d "$data_dir" ] && [ -w "$data_dir" ]; then
        log "Data directory: OK"
    else
        log "Data directory: NOT ACCESSIBLE"
        return 1
    fi
    
    return 0
}

# Memory usage check
check_memory_usage() {
    log "Checking memory usage..."
    
    if command -v free >/dev/null 2>&1; then
        local memory_usage=$(free | awk 'NR==2{printf "%.1f", $3*100/$2}')
        log "Memory usage: ${memory_usage}%"
        
        # Warning if memory usage is above 90%
        if [ "$(echo "$memory_usage > 90" | bc -l 2>/dev/null || echo 0)" -eq 1 ]; then
            log "WARNING: High memory usage detected"
        fi
    else
        log "Memory check not available"
    fi
}

# Disk usage check
check_disk_usage() {
    log "Checking disk usage..."
    
    local data_dir="${DATA_DIR:-/app/data}"
    local log_dir="${LOG_DIR:-/app/logs}"
    
    for dir in "$data_dir" "$log_dir"; do
        if [ -d "$dir" ]; then
            local usage=$(df "$dir" | awk 'NR==2 {print $5}' | sed 's/%//')
            log "Disk usage for $dir: ${usage}%"
            
            # Warning if disk usage is above 85%
            if [ "$usage" -gt 85 ]; then
                log "WARNING: High disk usage in $dir"
            fi
        fi
    done
}

# Main health check function
main() {
    log "Starting health check..."
    
    local retries=0
    local success=false
    
    while [ $retries -lt $MAX_RETRIES ]; do
        log "Health check attempt $((retries + 1))/$MAX_RETRIES"
        
        # Primary health check - HTTP endpoint
        if check_health "$HEALTH_ENDPOINT" "$TIMEOUT"; then
            if [ -n "$response" ]; then
                if parse_health_response "$response"; then
                    log "HTTP health check: ${GREEN}PASSED${NC}"
                    success=true
                    break
                else
                    log "HTTP health check: ${RED}FAILED${NC} (unhealthy response)"
                fi
            else
                log "HTTP health check: ${GREEN}PASSED${NC} (empty response)"
                success=true
                break
            fi
        else
            log "HTTP health check: ${RED}FAILED${NC}"
        fi
        
        retries=$((retries + 1))
        if [ $retries -lt $MAX_RETRIES ]; then
            log "Retrying in ${RETRY_DELAY} seconds..."
            sleep $RETRY_DELAY
        fi
    done
    
    if [ "$success" = false ]; then
        log "${RED}Primary health check failed after $MAX_RETRIES attempts${NC}"
        exit 1
    fi
    
    # Additional health checks (non-blocking)
    log "Performing additional health checks..."
    
    # These checks are informational and don't fail the health check
    check_database || log "Database check failed (non-blocking)"
    check_redis || log "Redis check failed (non-blocking)"
    check_application_health || log "Application health check failed (non-blocking)"
    
    # Resource checks (informational only)
    check_memory_usage
    check_disk_usage
    
    log "${GREEN}Health check completed successfully${NC}"
    exit 0
}

# Handle script arguments
case "${1:-}" in
    --help|-h)
        echo "Usage: $0 [options]"
        echo "Options:"
        echo "  --help, -h          Show this help message"
        echo "  --endpoint URL      Health check endpoint (default: $HEALTH_ENDPOINT)"
        echo "  --timeout SECONDS   Request timeout (default: $TIMEOUT)"
        echo "  --retries NUMBER    Maximum retries (default: $MAX_RETRIES)"
        echo ""
        echo "Environment variables:"
        echo "  HEALTH_ENDPOINT     Health check endpoint URL"
        echo "  TIMEOUT             Request timeout in seconds"
        echo "  MAX_RETRIES         Maximum number of retries"
        echo "  RETRY_DELAY         Delay between retries in seconds"
        echo "  DATABASE_HOST       Database host for connectivity check"
        echo "  DATABASE_PORT       Database port for connectivity check"
        echo "  REDIS_HOST          Redis host for connectivity check"
        echo "  REDIS_PORT          Redis port for connectivity check"
        exit 0
        ;;
    --endpoint)
        HEALTH_ENDPOINT="$2"
        shift 2
        ;;
    --timeout)
        TIMEOUT="$2"
        shift 2
        ;;
    --retries)
        MAX_RETRIES="$2"
        shift 2
        ;;
    *)
        # No arguments, proceed with health check
        ;;
esac

# Run the main health check
main "$@"