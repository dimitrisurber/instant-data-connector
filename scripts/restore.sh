#!/bin/bash
# Database restore script for PostgreSQL FDW Data Connector
set -euo pipefail

# Script configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
BACKUP_DIR="${BACKUP_DIR:-/backups/postgres}"
LOG_FILE="/tmp/restore-$(date +%Y%m%d-%H%M%S).log"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Default configuration
DB_HOST="${POSTGRES_HOST:-localhost}"
DB_PORT="${POSTGRES_PORT:-5432}"
DB_NAME="${POSTGRES_DB:-instant_connector}"
DB_USER="${POSTGRES_USER:-connector_user}"
DB_PASSWORD="${POSTGRES_PASSWORD:-}"
BACKUP_FILE=""
RESTORE_TYPE="full"
CREATE_DATABASE="${CREATE_DATABASE:-false}"
DROP_EXISTING="${DROP_EXISTING:-false}"
FORCE_RESTORE="${FORCE_RESTORE:-false}"
ENCRYPTION_PASSWORD="${ENCRYPTION_PASSWORD:-}"
S3_BUCKET="${S3_BUCKET:-}"
AWS_ACCESS_KEY_ID="${AWS_ACCESS_KEY_ID:-}"
AWS_SECRET_ACCESS_KEY="${AWS_SECRET_ACCESS_KEY:-}"

# Logging function
log() {
    local level="$1"
    shift
    local message="$*"
    local timestamp="$(date '+%Y-%m-%d %H:%M:%S')"
    
    case "$level" in
        INFO)  echo -e "${GREEN}[INFO]${NC} $message" | tee -a "$LOG_FILE" ;;
        WARN)  echo -e "${YELLOW}[WARN]${NC} $message" | tee -a "$LOG_FILE" ;;
        ERROR) echo -e "${RED}[ERROR]${NC} $message" | tee -a "$LOG_FILE" ;;
        DEBUG) echo -e "${BLUE}[DEBUG]${NC} $message" | tee -a "$LOG_FILE" ;;
        *)     echo -e "[$level] $message" | tee -a "$LOG_FILE" ;;
    esac
}

# Help function
show_help() {
    cat << EOF
PostgreSQL Database Restore Script

Usage: $(basename "$0") [OPTIONS]

OPTIONS:
    --host HOST                Database host [default: localhost]
    --port PORT                Database port [default: 5432]
    --database DB              Target database name [default: instant_connector]
    --user USER                Database user [default: connector_user]
    --password PASSWORD        Database password
    --backup-file FILE         Backup file to restore (required)
    --backup-dir DIR           Backup directory [default: /backups/postgres]
    --s3-bucket BUCKET         Download backup from S3 bucket
    --s3-key KEY               S3 object key for backup file
    --create-database          Create target database if it doesn't exist
    --drop-existing            Drop existing database before restore
    --force                    Force restore without confirmation prompts
    --decrypt                  Decrypt backup file before restore
    --list-backups             List available backup files
    --restore-schema-only      Restore schema only
    --restore-data-only        Restore data only
    --exclude-table-data TABLE Exclude data from specific table
    --jobs N                   Use N parallel jobs for restore [default: 1]
    -h, --help                 Show this help message

EXAMPLES:
    # List available backups
    $(basename "$0") --list-backups

    # Restore from local backup file
    $(basename "$0") --backup-file /backups/postgres/mydb_20240101_120000.sql.gz --password mypassword

    # Restore from S3
    $(basename "$0") --s3-bucket my-backup-bucket --s3-key postgres-backups/mydb_20240101_120000.sql.gz --password mypassword

    # Create new database and restore
    $(basename "$0") --backup-file mydb_backup.sql --create-database --password mypassword

    # Force restore with database recreation
    $(basename "$0") --backup-file mydb_backup.sql --drop-existing --create-database --force --password mypassword

ENVIRONMENT VARIABLES:
    POSTGRES_HOST              Database host
    POSTGRES_PORT              Database port
    POSTGRES_DB                Database name
    POSTGRES_USER              Database user
    POSTGRES_PASSWORD          Database password
    BACKUP_DIR                 Backup directory
    S3_BUCKET                  S3 bucket for backup downloads
    AWS_ACCESS_KEY_ID          AWS access key ID
    AWS_SECRET_ACCESS_KEY      AWS secret access key
    ENCRYPTION_PASSWORD        Password for backup decryption

EOF
}

# Parse command line arguments
parse_args() {
    PARALLEL_JOBS=1
    EXCLUDE_TABLE_DATA=""
    LIST_BACKUPS=false
    S3_KEY=""
    
    while [[ $# -gt 0 ]]; do
        case $1 in
            --host)
                DB_HOST="$2"
                shift 2
                ;;
            --port)
                DB_PORT="$2"
                shift 2
                ;;
            --database)
                DB_NAME="$2"
                shift 2
                ;;
            --user)
                DB_USER="$2"
                shift 2
                ;;
            --password)
                DB_PASSWORD="$2"
                shift 2
                ;;
            --backup-file)
                BACKUP_FILE="$2"
                shift 2
                ;;
            --backup-dir)
                BACKUP_DIR="$2"
                shift 2
                ;;
            --s3-bucket)
                S3_BUCKET="$2"
                shift 2
                ;;
            --s3-key)
                S3_KEY="$2"
                shift 2
                ;;
            --create-database)
                CREATE_DATABASE="true"
                shift
                ;;
            --drop-existing)
                DROP_EXISTING="true"
                shift
                ;;
            --force)
                FORCE_RESTORE="true"
                shift
                ;;
            --decrypt)
                DECRYPT_BACKUP="true"
                shift
                ;;
            --list-backups)
                LIST_BACKUPS="true"
                shift
                ;;
            --restore-schema-only)
                RESTORE_TYPE="schema"
                shift
                ;;
            --restore-data-only)
                RESTORE_TYPE="data"
                shift
                ;;
            --exclude-table-data)
                EXCLUDE_TABLE_DATA="$2"
                shift 2
                ;;
            --jobs)
                PARALLEL_JOBS="$2"
                shift 2
                ;;
            -h|--help)
                show_help
                exit 0
                ;;
            *)
                log ERROR "Unknown option: $1"
                show_help
                exit 1
                ;;
        esac
    done
}

# List available backup files
list_backups() {
    log INFO "Available backup files:"
    
    # List local backups
    if [[ -d "$BACKUP_DIR" ]]; then
        log INFO "Local backups in $BACKUP_DIR:"
        find "$BACKUP_DIR" -name "${DB_NAME}_*.sql*" -type f -printf "%T@ %Tc %p\n" 2>/dev/null | sort -nr | head -20 | while read -r timestamp date time file; do
            local size=$(du -h "$file" | cut -f1)
            echo "  $(basename "$file") - $date $time ($size)"
        done
    fi
    
    # List S3 backups if configured
    if [[ -n "$S3_BUCKET" ]] && command -v aws >/dev/null; then
        log INFO "S3 backups in s3://$S3_BUCKET/postgres-backups/:"
        aws s3 ls "s3://$S3_BUCKET/postgres-backups/" --human-readable --summarize | grep "${DB_NAME}_" | head -20 || log WARN "No S3 backups found or AWS CLI not configured"
    fi
}

# Validate configuration
validate_config() {
    log INFO "Validating restore configuration..."
    
    # Check if PostgreSQL client tools are available
    if ! command -v psql >/dev/null; then
        log ERROR "psql not found. Please install PostgreSQL client tools"
        exit 1
    fi
    
    if ! command -v pg_restore >/dev/null; then
        log ERROR "pg_restore not found. Please install PostgreSQL client tools"
        exit 1
    fi
    
    # Handle list backups request
    if [[ "$LIST_BACKUPS" == "true" ]]; then
        list_backups
        exit 0
    fi
    
    # Check backup file or S3 configuration
    if [[ -z "$BACKUP_FILE" && -z "$S3_KEY" ]]; then
        log ERROR "No backup file specified. Use --backup-file or --s3-key"
        exit 1
    fi
    
    # Check database password
    if [[ -z "$DB_PASSWORD" ]]; then
        if [[ "$FORCE_RESTORE" == "false" ]]; then
            echo -n "Enter database password: "
            read -s DB_PASSWORD
            echo
        else
            log ERROR "Database password not provided"
            exit 1
        fi
    fi
    
    # Check S3 configuration if S3 restore is requested
    if [[ -n "$S3_KEY" || -n "$S3_BUCKET" ]]; then
        if ! command -v aws >/dev/null; then
            log ERROR "AWS CLI not found. Please install AWS CLI for S3 downloads"
            exit 1
        fi
        
        if [[ -z "$S3_BUCKET" ]]; then
            log ERROR "S3 bucket not specified"
            exit 1
        fi
        
        if [[ -z "$S3_KEY" ]]; then
            log ERROR "S3 key not specified"
            exit 1
        fi
    fi
    
    # Check decryption configuration
    if [[ "${DECRYPT_BACKUP:-false}" == "true" ]]; then
        if ! command -v gpg >/dev/null; then
            log ERROR "GPG not found. Please install GPG for backup decryption"
            exit 1
        fi
        
        if [[ -z "$ENCRYPTION_PASSWORD" ]]; then
            if [[ "$FORCE_RESTORE" == "false" ]]; then
                echo -n "Enter decryption password: "
                read -s ENCRYPTION_PASSWORD
                echo
            else
                log ERROR "Decryption password not provided"
                exit 1
            fi
        fi
    fi
    
    log INFO "Configuration validation completed"
}

# Test database connection
test_connection() {
    log INFO "Testing database connection..."
    
    export PGPASSWORD="$DB_PASSWORD"
    
    if ! pg_isready -h "$DB_HOST" -p "$DB_PORT" -U "$DB_USER" >/dev/null 2>&1; then
        log ERROR "Cannot connect to PostgreSQL server"
        log ERROR "Host: $DB_HOST:$DB_PORT, User: $DB_USER"
        exit 1
    fi
    
    log INFO "Database server connection successful"
}

# Check if database exists
check_database_exists() {
    export PGPASSWORD="$DB_PASSWORD"
    
    local db_exists=$(psql -h "$DB_HOST" -p "$DB_PORT" -U "$DB_USER" -d postgres -t -c "SELECT 1 FROM pg_database WHERE datname='$DB_NAME';" 2>/dev/null | xargs)
    
    if [[ "$db_exists" == "1" ]]; then
        return 0
    else
        return 1
    fi
}

# Create database
create_database() {
    log INFO "Creating database: $DB_NAME"
    
    export PGPASSWORD="$DB_PASSWORD"
    
    if psql -h "$DB_HOST" -p "$DB_PORT" -U "$DB_USER" -d postgres -c "CREATE DATABASE \"$DB_NAME\";" >/dev/null 2>&1; then
        log INFO "Database created successfully"
    else
        log ERROR "Failed to create database"
        exit 1
    fi
}

# Drop database
drop_database() {
    log INFO "Dropping existing database: $DB_NAME"
    
    export PGPASSWORD="$DB_PASSWORD"
    
    # Terminate existing connections
    psql -h "$DB_HOST" -p "$DB_PORT" -U "$DB_USER" -d postgres -c "
        SELECT pg_terminate_backend(pid) 
        FROM pg_stat_activity 
        WHERE datname = '$DB_NAME' AND pid <> pg_backend_pid();
    " >/dev/null 2>&1 || true
    
    if psql -h "$DB_HOST" -p "$DB_PORT" -U "$DB_USER" -d postgres -c "DROP DATABASE IF EXISTS \"$DB_NAME\";" >/dev/null 2>&1; then
        log INFO "Database dropped successfully"
    else
        log ERROR "Failed to drop database"
        exit 1
    fi
}

# Download backup from S3
download_from_s3() {
    local local_file="$BACKUP_DIR/$(basename "$S3_KEY")"
    
    log INFO "Downloading backup from S3: s3://$S3_BUCKET/$S3_KEY"
    
    # Create backup directory if it doesn't exist
    mkdir -p "$BACKUP_DIR"
    
    if aws s3 cp "s3://$S3_BUCKET/$S3_KEY" "$local_file"; then
        log INFO "S3 download completed: $local_file"
        BACKUP_FILE="$local_file"
    else
        log ERROR "S3 download failed"
        exit 1
    fi
}

# Prepare backup file
prepare_backup_file() {
    log INFO "Preparing backup file for restore..."
    
    # Download from S3 if needed
    if [[ -n "$S3_KEY" ]]; then
        download_from_s3
    fi
    
    # Check if backup file exists
    if [[ ! -f "$BACKUP_FILE" ]]; then
        log ERROR "Backup file not found: $BACKUP_FILE"
        exit 1
    fi
    
    # Check if backup file is empty
    if [[ ! -s "$BACKUP_FILE" ]]; then
        log ERROR "Backup file is empty: $BACKUP_FILE"
        exit 1
    fi
    
    local original_file="$BACKUP_FILE"
    
    # Decrypt if needed
    if [[ "$BACKUP_FILE" == *.gpg ]] || [[ "${DECRYPT_BACKUP:-false}" == "true" ]]; then
        log INFO "Decrypting backup file..."
        local decrypted_file="${BACKUP_FILE%.gpg}"
        
        if echo "$ENCRYPTION_PASSWORD" | gpg --batch --yes --passphrase-fd 0 --decrypt "$BACKUP_FILE" > "$decrypted_file"; then
            log INFO "Backup decryption completed"
            BACKUP_FILE="$decrypted_file"
        else
            log ERROR "Backup decryption failed"
            exit 1
        fi
    fi
    
    # Decompress if needed
    if [[ "$BACKUP_FILE" == *.gz ]]; then
        log INFO "Decompressing backup file..."
        local decompressed_file="${BACKUP_FILE%.gz}"
        
        if gunzip -c "$BACKUP_FILE" > "$decompressed_file"; then
            log INFO "Backup decompression completed"
            BACKUP_FILE="$decompressed_file"
        else
            log ERROR "Backup decompression failed"
            exit 1
        fi
    fi
    
    # Show backup file information
    local backup_size=$(du -h "$BACKUP_FILE" | cut -f1)
    log INFO "Backup file ready: $(basename "$BACKUP_FILE") ($backup_size)"
    
    # Verify backup file format
    if head -n 5 "$BACKUP_FILE" | grep -q "PostgreSQL database dump"; then
        log INFO "Backup file format verification passed"
    else
        log WARN "Could not verify backup file format"
    fi
}

# Get database information before restore
get_pre_restore_info() {
    log INFO "Getting database information before restore..."
    
    export PGPASSWORD="$DB_PASSWORD"
    
    if check_database_exists; then
        # Get table count
        local table_count=$(psql -h "$DB_HOST" -p "$DB_PORT" -U "$DB_USER" -d "$DB_NAME" -t -c "
            SELECT count(*) FROM information_schema.tables 
            WHERE table_schema = 'public' AND table_type = 'BASE TABLE';
        " 2>/dev/null | xargs || echo "0")
        
        # Get database size
        local db_size=$(psql -h "$DB_HOST" -p "$DB_PORT" -U "$DB_USER" -d "$DB_NAME" -t -c "
            SELECT pg_size_pretty(pg_database_size('$DB_NAME'));
        " 2>/dev/null | xargs || echo "unknown")
        
        log INFO "Current database state:"
        log INFO "  - Tables: $table_count"
        log INFO "  - Size: $db_size"
        
        # Show list of tables
        log INFO "Existing tables:"
        psql -h "$DB_HOST" -p "$DB_PORT" -U "$DB_USER" -d "$DB_NAME" -c "
            SELECT schemaname, tablename, 
                   pg_size_pretty(pg_total_relation_size(schemaname||'.'||tablename)) as size
            FROM pg_tables 
            WHERE schemaname = 'public' 
            ORDER BY pg_total_relation_size(schemaname||'.'||tablename) DESC 
            LIMIT 10;
        " 2>/dev/null || log WARN "Could not retrieve table information"
    else
        log INFO "Database does not exist yet"
    fi
}

# Confirm restore operation
confirm_restore() {
    if [[ "$FORCE_RESTORE" == "true" ]]; then
        log INFO "Force restore enabled, skipping confirmation"
        return 0
    fi
    
    log WARN "This operation will restore database '$DB_NAME' from backup file:"
    log WARN "  Backup: $(basename "$BACKUP_FILE")"
    log WARN "  Host: $DB_HOST:$DB_PORT"
    log WARN "  Type: $RESTORE_TYPE restore"
    
    if [[ "$DROP_EXISTING" == "true" ]]; then
        log WARN "  WARNING: Existing database will be DROPPED!"
    fi
    
    if [[ "$CREATE_DATABASE" == "true" ]]; then
        log WARN "  Database will be created if it doesn't exist"
    fi
    
    echo -n "Are you sure you want to continue? (yes/no): "
    read -r confirmation
    
    case "$confirmation" in
        yes|YES|y|Y)
            log INFO "Restore confirmed by user"
            ;;
        *)
            log INFO "Restore cancelled by user"
            exit 0
            ;;
    esac
}

# Perform database restore
perform_restore() {
    log INFO "Starting database restore..."
    
    export PGPASSWORD="$DB_PASSWORD"
    
    # Handle database creation/dropping
    if [[ "$DROP_EXISTING" == "true" ]]; then
        if check_database_exists; then
            drop_database
        fi
    fi
    
    if [[ "$CREATE_DATABASE" == "true" ]]; then
        if ! check_database_exists; then
            create_database
        fi
    fi
    
    # Check if target database exists
    if ! check_database_exists; then
        log ERROR "Target database '$DB_NAME' does not exist"
        log ERROR "Use --create-database to create it automatically"
        exit 1
    fi
    
    # Build restore command
    local restore_cmd=(
        "psql"
        "-h" "$DB_HOST"
        "-p" "$DB_PORT"
        "-U" "$DB_USER"
        "-d" "$DB_NAME"
        "-v" "ON_ERROR_STOP=1"
        "--no-password"
    )
    
    # Add restore type specific options
    case "$RESTORE_TYPE" in
        schema)
            # For schema-only restore, we need to filter the SQL
            log INFO "Performing schema-only restore..."
            if grep -v "^COPY\|^INSERT\|^\\\\\\." "$BACKUP_FILE" | "${restore_cmd[@]}" 2>"$LOG_FILE.restore"; then
                log INFO "Schema restore completed successfully"
            else
                log ERROR "Schema restore failed"
                cat "$LOG_FILE.restore" | tail -20 >> "$LOG_FILE"
                exit 1
            fi
            ;;
        data)
            # For data-only restore, we need to filter the SQL
            log INFO "Performing data-only restore..."
            if grep "^COPY\|^INSERT\|^\\\\\\." "$BACKUP_FILE" | "${restore_cmd[@]}" 2>"$LOG_FILE.restore"; then
                log INFO "Data restore completed successfully"
            else
                log ERROR "Data restore failed"
                cat "$LOG_FILE.restore" | tail -20 >> "$LOG_FILE"
                exit 1
            fi
            ;;
        *)
            # Full restore
            log INFO "Performing full database restore..."
            if "${restore_cmd[@]}" < "$BACKUP_FILE" 2>"$LOG_FILE.restore"; then
                log INFO "Database restore completed successfully"
            else
                log ERROR "Database restore failed"
                cat "$LOG_FILE.restore" | tail -20 >> "$LOG_FILE"
                exit 1
            fi
            ;;
    esac
}

# Get database information after restore
get_post_restore_info() {
    log INFO "Getting database information after restore..."
    
    export PGPASSWORD="$DB_PASSWORD"
    
    # Get table count
    local table_count=$(psql -h "$DB_HOST" -p "$DB_PORT" -U "$DB_USER" -d "$DB_NAME" -t -c "
        SELECT count(*) FROM information_schema.tables 
        WHERE table_schema = 'public' AND table_type = 'BASE TABLE';
    " 2>/dev/null | xargs || echo "0")
    
    # Get database size
    local db_size=$(psql -h "$DB_HOST" -p "$DB_PORT" -U "$DB_USER" -d "$DB_NAME" -t -c "
        SELECT pg_size_pretty(pg_database_size('$DB_NAME'));
    " 2>/dev/null | xargs || echo "unknown")
    
    # Get row counts for major tables
    log INFO "Restored database state:"
    log INFO "  - Tables: $table_count"
    log INFO "  - Size: $db_size"
    
    log INFO "Table row counts:"
    psql -h "$DB_HOST" -p "$DB_PORT" -U "$DB_USER" -d "$DB_NAME" -c "
        SELECT 
            schemaname,
            tablename,
            n_tup_ins as rows,
            pg_size_pretty(pg_total_relation_size(schemaname||'.'||tablename)) as size
        FROM pg_stat_user_tables 
        WHERE schemaname = 'public'
        ORDER BY n_tup_ins DESC 
        LIMIT 10;
    " 2>/dev/null || log WARN "Could not retrieve table statistics"
}

# Verify restore integrity
verify_restore() {
    log INFO "Verifying restore integrity..."
    
    export PGPASSWORD="$DB_PASSWORD"
    
    # Basic connectivity test
    if ! psql -h "$DB_HOST" -p "$DB_PORT" -U "$DB_USER" -d "$DB_NAME" -c "SELECT 1;" >/dev/null 2>&1; then
        log ERROR "Cannot connect to restored database"
        exit 1
    fi
    
    # Check for basic tables (if we know what to expect)
    local essential_tables=("users" "orders" "products")
    local missing_tables=()
    
    for table in "${essential_tables[@]}"; do
        local exists=$(psql -h "$DB_HOST" -p "$DB_PORT" -U "$DB_USER" -d "$DB_NAME" -t -c "
            SELECT 1 FROM information_schema.tables 
            WHERE table_name = '$table' AND table_schema = 'public';
        " 2>/dev/null | xargs || echo "")
        
        if [[ "$exists" != "1" ]]; then
            missing_tables+=("$table")
        fi
    done
    
    if [[ ${#missing_tables[@]} -gt 0 ]]; then
        log WARN "Some expected tables are missing: ${missing_tables[*]}"
        log WARN "This might be normal depending on your backup content"
    else
        log INFO "Essential tables verification passed"
    fi
    
    # Check for foreign key constraints
    local fk_count=$(psql -h "$DB_HOST" -p "$DB_PORT" -U "$DB_USER" -d "$DB_NAME" -t -c "
        SELECT count(*) FROM information_schema.table_constraints 
        WHERE constraint_type = 'FOREIGN KEY';
    " 2>/dev/null | xargs || echo "0")
    
    log INFO "Foreign key constraints: $fk_count"
    
    # Check for indexes
    local index_count=$(psql -h "$DB_HOST" -p "$DB_PORT" -U "$DB_USER" -d "$DB_NAME" -t -c "
        SELECT count(*) FROM pg_indexes WHERE schemaname = 'public';
    " 2>/dev/null | xargs || echo "0")
    
    log INFO "Indexes: $index_count"
    
    log INFO "Restore integrity verification completed"
}

# Cleanup temporary files
cleanup_temp_files() {
    log INFO "Cleaning up temporary files..."
    
    # Remove decompressed/decrypted files if they were created
    if [[ -n "${BACKUP_FILE:-}" && "$BACKUP_FILE" != "$original_file" ]]; then
        if [[ -f "$BACKUP_FILE" ]]; then
            rm -f "$BACKUP_FILE"
            log INFO "Removed temporary file: $(basename "$BACKUP_FILE")"
        fi
    fi
    
    # Remove S3 downloaded file if it was temporary
    if [[ -n "$S3_KEY" && -f "$BACKUP_DIR/$(basename "$S3_KEY")" ]]; then
        rm -f "$BACKUP_DIR/$(basename "$S3_KEY")"
        log INFO "Removed S3 download file"
    fi
    
    # Remove restore log file
    rm -f "$LOG_FILE.restore"
}

# Generate restore report
generate_report() {
    local report_file="$BACKUP_DIR/restore-report-$(date +%Y%m%d-%H%M%S).txt"
    
    cat > "$report_file" << EOF
PostgreSQL Database Restore Report
Generated: $(date '+%Y-%m-%d %H:%M:%S')

Database Information:
- Host: $DB_HOST:$DB_PORT
- Database: $DB_NAME
- User: $DB_USER

Restore Information:
- Type: $RESTORE_TYPE
- Source File: $(basename "${original_file:-$BACKUP_FILE}")
- Target Database: $DB_NAME
- Create Database: $CREATE_DATABASE
- Drop Existing: $DROP_EXISTING

Options:
- Force Restore: $FORCE_RESTORE
- Parallel Jobs: $PARALLEL_JOBS
$(if [[ -n "$EXCLUDE_TABLE_DATA" ]]; then echo "- Excluded Table Data: $EXCLUDE_TABLE_DATA"; fi)

$(if [[ -n "$S3_BUCKET" ]]; then echo "S3 Information:
- Bucket: $S3_BUCKET
- Key: $S3_KEY"; fi)

Log File: $LOG_FILE
EOF
    
    log INFO "Restore report generated: $report_file"
}

# Send notification
send_notification() {
    local status="$1"
    
    # Email notification (if configured)
    if command -v mail >/dev/null && [[ -n "${NOTIFICATION_EMAIL:-}" ]]; then
        local subject="PostgreSQL Restore $status - $DB_NAME"
        local body="Database restore $status for $DB_NAME on $DB_HOST

Backup file: $(basename "${original_file:-$BACKUP_FILE}")
Restore type: $RESTORE_TYPE
Timestamp: $(date '+%Y-%m-%d %H:%M:%S')
Log file: $LOG_FILE"
        
        echo "$body" | mail -s "$subject" "$NOTIFICATION_EMAIL"
        log INFO "Email notification sent to $NOTIFICATION_EMAIL"
    fi
}

# Cleanup function
cleanup() {
    local exit_code=$?
    
    # Clean up temporary files
    cleanup_temp_files
    
    # Clear password from environment
    unset PGPASSWORD
    
    if [[ $exit_code -eq 0 ]]; then
        log INFO "Restore process completed successfully"
        send_notification "SUCCESS"
    else
        log ERROR "Restore process failed with exit code: $exit_code"
        send_notification "FAILED"
    fi
    
    log INFO "Restore log saved to: $LOG_FILE"
    
    exit $exit_code
}

# Main restore function
main() {
    log INFO "Starting PostgreSQL restore process..."
    log INFO "Restore log: $LOG_FILE"
    
    # Parse command line arguments
    parse_args "$@"
    
    # Validate configuration
    validate_config
    
    # Test database connection
    test_connection
    
    # Prepare backup file
    prepare_backup_file
    
    # Get pre-restore information
    get_pre_restore_info
    
    # Confirm restore operation
    confirm_restore
    
    # Perform restore
    perform_restore
    
    # Get post-restore information
    get_post_restore_info
    
    # Verify restore integrity
    verify_restore
    
    # Generate report
    generate_report
    
    log INFO "Database restore completed successfully"
}

# Set trap for cleanup
trap cleanup EXIT

# Run main function with all arguments
main "$@"