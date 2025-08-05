#!/bin/bash
# Database backup script for PostgreSQL FDW Data Connector
set -euo pipefail

# Script configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
BACKUP_DIR="${BACKUP_DIR:-/backups/postgres}"
LOG_FILE="/tmp/backup-$(date +%Y%m%d-%H%M%S).log"

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
BACKUP_RETENTION_DAYS="${BACKUP_RETENTION_DAYS:-30}"
COMPRESSION_LEVEL="${COMPRESSION_LEVEL:-6}"
AUTO_BACKUP="${AUTO_BACKUP:-false}"
S3_BUCKET="${S3_BUCKET:-}"
AWS_ACCESS_KEY_ID="${AWS_ACCESS_KEY_ID:-}"
AWS_SECRET_ACCESS_KEY="${AWS_SECRET_ACCESS_KEY:-}"
ENCRYPTION_ENABLED="${ENCRYPTION_ENABLED:-true}"
ENCRYPTION_PASSWORD="${ENCRYPTION_PASSWORD:-}"

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
PostgreSQL Database Backup Script

Usage: $(basename "$0") [OPTIONS]

OPTIONS:
    --host HOST                Database host [default: localhost]
    --port PORT                Database port [default: 5432]
    --database DB              Database name [default: instant_connector]
    --user USER                Database user [default: connector_user]
    --password PASSWORD        Database password
    --backup-dir DIR           Backup directory [default: /backups/postgres]
    --retention-days DAYS      Backup retention in days [default: 30]
    --compression-level LEVEL  Gzip compression level 1-9 [default: 6]
    --auto-backup              Run in auto-backup mode (non-interactive)
    --encrypt                  Encrypt backup files
    --s3-bucket BUCKET         Upload backups to S3 bucket
    --full-backup              Perform full database backup (default)
    --schema-only              Backup schema only
    --data-only                Backup data only
    --tables TABLES            Backup specific tables (comma-separated)
    --exclude-tables TABLES    Exclude specific tables (comma-separated)
    -h, --help                 Show this help message

EXAMPLES:
    # Basic backup
    $(basename "$0") --password mypassword

    # Backup with S3 upload
    $(basename "$0") --password mypassword --s3-bucket my-backup-bucket

    # Schema-only backup
    $(basename "$0") --password mypassword --schema-only

    # Backup specific tables
    $(basename "$0") --password mypassword --tables users,orders,products

ENVIRONMENT VARIABLES:
    POSTGRES_HOST              Database host
    POSTGRES_PORT              Database port
    POSTGRES_DB                Database name
    POSTGRES_USER              Database user
    POSTGRES_PASSWORD          Database password
    BACKUP_DIR                 Backup directory
    BACKUP_RETENTION_DAYS      Backup retention days
    S3_BUCKET                  S3 bucket for backup uploads
    AWS_ACCESS_KEY_ID          AWS access key ID
    AWS_SECRET_ACCESS_KEY      AWS secret access key
    ENCRYPTION_PASSWORD        Password for backup encryption

EOF
}

# Parse command line arguments
parse_args() {
    BACKUP_TYPE="full"
    SPECIFIC_TABLES=""
    EXCLUDE_TABLES=""
    
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
            --backup-dir)
                BACKUP_DIR="$2"
                shift 2
                ;;
            --retention-days)
                BACKUP_RETENTION_DAYS="$2"
                shift 2
                ;;
            --compression-level)
                COMPRESSION_LEVEL="$2"
                shift 2
                ;;
            --auto-backup)
                AUTO_BACKUP="true"
                shift
                ;;
            --encrypt)
                ENCRYPTION_ENABLED="true"
                shift
                ;;
            --s3-bucket)
                S3_BUCKET="$2"
                shift 2
                ;;
            --full-backup)
                BACKUP_TYPE="full"
                shift
                ;;
            --schema-only)
                BACKUP_TYPE="schema"
                shift
                ;;
            --data-only)
                BACKUP_TYPE="data"
                shift
                ;;
            --tables)
                SPECIFIC_TABLES="$2"
                BACKUP_TYPE="tables"
                shift 2
                ;;
            --exclude-tables)
                EXCLUDE_TABLES="$2"
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

# Validate configuration
validate_config() {
    log INFO "Validating backup configuration..."
    
    # Check if PostgreSQL client tools are available
    if ! command -v pg_dump >/dev/null; then
        log ERROR "pg_dump not found. Please install PostgreSQL client tools"
        exit 1
    fi
    
    # Check database password
    if [[ -z "$DB_PASSWORD" ]]; then
        if [[ "$AUTO_BACKUP" == "false" ]]; then
            echo -n "Enter database password: "
            read -s DB_PASSWORD
            echo
        else
            log ERROR "Database password not provided"
            exit 1
        fi
    fi
    
    # Create backup directory
    if [[ ! -d "$BACKUP_DIR" ]]; then
        log INFO "Creating backup directory: $BACKUP_DIR"
        mkdir -p "$BACKUP_DIR"
    fi
    
    # Check S3 configuration if S3 upload is enabled
    if [[ -n "$S3_BUCKET" ]]; then
        if ! command -v aws >/dev/null; then
            log ERROR "AWS CLI not found. Please install AWS CLI for S3 uploads"
            exit 1
        fi
        
        if [[ -z "$AWS_ACCESS_KEY_ID" || -z "$AWS_SECRET_ACCESS_KEY" ]]; then
            log WARN "AWS credentials not found in environment variables"
            log WARN "Make sure AWS CLI is configured or credentials are available"
        fi
    fi
    
    # Check encryption configuration
    if [[ "$ENCRYPTION_ENABLED" == "true" ]]; then
        if ! command -v gpg >/dev/null; then
            log ERROR "GPG not found. Please install GPG for backup encryption"
            exit 1
        fi
        
        if [[ -z "$ENCRYPTION_PASSWORD" ]]; then
            if [[ "$AUTO_BACKUP" == "false" ]]; then
                echo -n "Enter encryption password: "
                read -s ENCRYPTION_PASSWORD
                echo
            else
                log ERROR "Encryption password not provided"
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
    
    if ! pg_isready -h "$DB_HOST" -p "$DB_PORT" -U "$DB_USER" -d "$DB_NAME" >/dev/null 2>&1; then
        log ERROR "Cannot connect to database"
        log ERROR "Host: $DB_HOST:$DB_PORT, Database: $DB_NAME, User: $DB_USER"
        exit 1
    fi
    
    log INFO "Database connection successful"
}

# Get database size
get_database_size() {
    export PGPASSWORD="$DB_PASSWORD"
    
    local size_query="SELECT pg_size_pretty(pg_database_size('$DB_NAME'));"
    local db_size=$(psql -h "$DB_HOST" -p "$DB_PORT" -U "$DB_USER" -d "$DB_NAME" -t -c "$size_query" 2>/dev/null | xargs)
    
    if [[ -n "$db_size" ]]; then
        log INFO "Database size: $db_size"
    else
        log WARN "Could not determine database size"
    fi
}

# Create backup filename
create_backup_filename() {
    local timestamp=$(date +%Y%m%d_%H%M%S)
    local suffix=""
    
    case "$BACKUP_TYPE" in
        schema) suffix="_schema" ;;
        data) suffix="_data" ;;
        tables) suffix="_tables" ;;
        *) suffix="" ;;
    esac
    
    echo "${DB_NAME}_${timestamp}${suffix}.sql"
}

# Perform database backup
perform_backup() {
    log INFO "Starting database backup..."
    
    local backup_filename=$(create_backup_filename)
    local backup_path="$BACKUP_DIR/$backup_filename"
    local compressed_path="${backup_path}.gz"
    local encrypted_path="${compressed_path}.gpg"
    
    export PGPASSWORD="$DB_PASSWORD"
    
    # Build pg_dump command
    local pg_dump_cmd=(
        "pg_dump"
        "-h" "$DB_HOST"
        "-p" "$DB_PORT"
        "-U" "$DB_USER"
        "-d" "$DB_NAME"
        "--verbose"
        "--no-password"
    )
    
    # Add backup type specific options
    case "$BACKUP_TYPE" in
        schema)
            pg_dump_cmd+=("--schema-only")
            ;;
        data)
            pg_dump_cmd+=("--data-only")
            ;;
        tables)
            if [[ -n "$SPECIFIC_TABLES" ]]; then
                IFS=',' read -ra TABLES <<< "$SPECIFIC_TABLES"
                for table in "${TABLES[@]}"; do
                    pg_dump_cmd+=("-t" "${table// /}")
                done
            fi
            ;;
    esac
    
    # Add exclude tables if specified
    if [[ -n "$EXCLUDE_TABLES" ]]; then
        IFS=',' read -ra EXCLUDE <<< "$EXCLUDE_TABLES"
        for table in "${EXCLUDE[@]}"; do
            pg_dump_cmd+=("-T" "${table// /}")
        done
    fi
    
    # Execute backup
    log INFO "Backup file: $backup_filename"
    log INFO "Running: ${pg_dump_cmd[*]}"
    
    if "${pg_dump_cmd[@]}" > "$backup_path" 2>"$LOG_FILE.pg_dump"; then
        log INFO "Database backup completed successfully"
        
        # Show backup file size
        local backup_size=$(du -h "$backup_path" | cut -f1)
        log INFO "Backup file size: $backup_size"
        
        # Compress backup
        log INFO "Compressing backup file..."
        if gzip -"$COMPRESSION_LEVEL" "$backup_path"; then
            local compressed_size=$(du -h "$compressed_path" | cut -f1)
            log INFO "Compressed backup size: $compressed_size"
            backup_path="$compressed_path"
        else
            log WARN "Backup compression failed, keeping uncompressed file"
        fi
        
        # Encrypt backup if enabled
        if [[ "$ENCRYPTION_ENABLED" == "true" ]]; then
            log INFO "Encrypting backup file..."
            if echo "$ENCRYPTION_PASSWORD" | gpg --batch --yes --passphrase-fd 0 --cipher-algo AES256 --compress-algo 1 --symmetric --output "$encrypted_path" "$backup_path"; then
                log INFO "Backup encryption completed"
                rm -f "$backup_path"
                backup_path="$encrypted_path"
            else
                log WARN "Backup encryption failed, keeping unencrypted file"
            fi
        fi
        
        # Upload to S3 if configured
        if [[ -n "$S3_BUCKET" ]]; then
            upload_to_s3 "$backup_path"
        fi
        
        # Verify backup integrity
        verify_backup "$backup_path"
        
        log INFO "Final backup file: $backup_path"
        
    else
        log ERROR "Database backup failed"
        cat "$LOG_FILE.pg_dump" | tail -20 >> "$LOG_FILE"
        exit 1
    fi
}

# Upload backup to S3
upload_to_s3() {
    local backup_file="$1"
    local s3_key="postgres-backups/$(basename "$backup_file")"
    
    log INFO "Uploading backup to S3: s3://$S3_BUCKET/$s3_key"
    
    if aws s3 cp "$backup_file" "s3://$S3_BUCKET/$s3_key" --storage-class STANDARD_IA; then
        log INFO "S3 upload completed successfully"
        
        # Set lifecycle policy if not exists
        set_s3_lifecycle_policy
    else
        log ERROR "S3 upload failed"
        exit 1
    fi
}

# Set S3 lifecycle policy
set_s3_lifecycle_policy() {
    local lifecycle_config="/tmp/s3-lifecycle-config.json"
    
    cat > "$lifecycle_config" << EOF
{
    "Rules": [
        {
            "ID": "PostgreSQLBackupLifecycle",
            "Status": "Enabled",
            "Filter": {
                "Prefix": "postgres-backups/"
            },
            "Transitions": [
                {
                    "Days": 30,
                    "StorageClass": "GLACIER"
                },
                {
                    "Days": 90,
                    "StorageClass": "DEEP_ARCHIVE"
                }
            ],
            "Expiration": {
                "Days": $((BACKUP_RETENTION_DAYS * 3))
            }
        }
    ]
}
EOF
    
    if aws s3api get-bucket-lifecycle-configuration --bucket "$S3_BUCKET" >/dev/null 2>&1; then
        log INFO "S3 lifecycle policy already exists"
    else
        log INFO "Setting S3 lifecycle policy..."
        if aws s3api put-bucket-lifecycle-configuration --bucket "$S3_BUCKET" --lifecycle-configuration "file://$lifecycle_config"; then
            log INFO "S3 lifecycle policy set successfully"
        else
            log WARN "Failed to set S3 lifecycle policy"
        fi
    fi
    
    rm -f "$lifecycle_config"
}

# Verify backup integrity
verify_backup() {
    local backup_file="$1"
    
    log INFO "Verifying backup integrity..."
    
    # Basic file checks
    if [[ ! -f "$backup_file" ]]; then
        log ERROR "Backup file not found: $backup_file"
        exit 1
    fi
    
    if [[ ! -s "$backup_file" ]]; then
        log ERROR "Backup file is empty: $backup_file"
        exit 1
    fi
    
    # Check file format based on extension
    case "$backup_file" in
        *.gz)
            if gzip -t "$backup_file"; then
                log INFO "Compressed backup file integrity check passed"
            else
                log ERROR "Compressed backup file is corrupted"
                exit 1
            fi
            ;;
        *.gpg)
            # For encrypted files, we can only check if it's a valid GPG file
            if gpg --list-packets "$backup_file" >/dev/null 2>&1; then
                log INFO "Encrypted backup file format check passed"
            else
                log ERROR "Encrypted backup file format check failed"
                exit 1
            fi
            ;;
        *.sql)
            # For SQL files, check if it contains SQL content
            if head -n 10 "$backup_file" | grep -q "PostgreSQL database dump"; then
                log INFO "SQL backup file format check passed"
            else
                log WARN "SQL backup file format check inconclusive"
            fi
            ;;
    esac
    
    log INFO "Backup integrity verification completed"
}

# Clean up old backups
cleanup_old_backups() {
    log INFO "Cleaning up backups older than $BACKUP_RETENTION_DAYS days..."
    
    local deleted_count=0
    
    # Find and delete old local backups
    while IFS= read -r -d '' file; do
        log INFO "Deleting old backup: $(basename "$file")"
        rm -f "$file"
        ((deleted_count++))
    done < <(find "$BACKUP_DIR" -type f -name "${DB_NAME}_*.sql*" -mtime +${BACKUP_RETENTION_DAYS} -print0 2>/dev/null || true)
    
    # Clean up old S3 backups if configured
    if [[ -n "$S3_BUCKET" ]]; then
        log INFO "Cleaning up old S3 backups..."
        local cutoff_date=$(date -d "${BACKUP_RETENTION_DAYS} days ago" +%Y-%m-%d)
        
        # List and delete old S3 objects
        aws s3api list-objects-v2 --bucket "$S3_BUCKET" --prefix "postgres-backups/" --query "Contents[?LastModified<='$cutoff_date'].Key" --output text | while read -r key; do
            if [[ -n "$key" && "$key" != "None" ]]; then
                log INFO "Deleting old S3 backup: $key"
                aws s3 rm "s3://$S3_BUCKET/$key"
                ((deleted_count++))
            fi
        done
    fi
    
    if [[ $deleted_count -gt 0 ]]; then
        log INFO "Deleted $deleted_count old backup files"
    else
        log INFO "No old backup files found to delete"
    fi
}

# Generate backup report
generate_report() {
    local backup_file="$1"
    local report_file="$BACKUP_DIR/backup-report-$(date +%Y%m%d).txt"
    
    cat > "$report_file" << EOF
PostgreSQL Database Backup Report
Generated: $(date '+%Y-%m-%d %H:%M:%S')

Database Information:
- Host: $DB_HOST:$DB_PORT
- Database: $DB_NAME
- User: $DB_USER

Backup Information:
- Type: $BACKUP_TYPE
- File: $(basename "$backup_file")
- Size: $(du -h "$backup_file" | cut -f1)
- Compression: $(if [[ "$backup_file" == *.gz* ]]; then echo "Enabled (level $COMPRESSION_LEVEL)"; else echo "Disabled"; fi)
- Encryption: $(if [[ "$ENCRYPTION_ENABLED" == "true" ]]; then echo "Enabled"; else echo "Disabled"; fi)

Storage Information:
- Local Path: $backup_file
$(if [[ -n "$S3_BUCKET" ]]; then echo "- S3 Bucket: s3://$S3_BUCKET/postgres-backups/$(basename "$backup_file")"; fi)

Settings:
- Retention Days: $BACKUP_RETENTION_DAYS
- Auto Backup: $AUTO_BACKUP

Log File: $LOG_FILE
EOF
    
    log INFO "Backup report generated: $report_file"
}

# Send notification
send_notification() {
    local status="$1"
    local backup_file="$2"
    
    # Email notification (if configured)
    if command -v mail >/dev/null && [[ -n "${NOTIFICATION_EMAIL:-}" ]]; then
        local subject="PostgreSQL Backup $status - $DB_NAME"
        local body="Backup $status for database $DB_NAME on $DB_HOST

Backup file: $(basename "$backup_file")
Timestamp: $(date '+%Y-%m-%d %H:%M:%S')
Log file: $LOG_FILE"
        
        echo "$body" | mail -s "$subject" "$NOTIFICATION_EMAIL"
        log INFO "Email notification sent to $NOTIFICATION_EMAIL"
    fi
    
    # Slack notification (if configured)
    if [[ -n "${SLACK_WEBHOOK_URL:-}" ]]; then
        local color
        case "$status" in
            "SUCCESS") color="good" ;;
            "FAILED") color="danger" ;;
            *) color="warning" ;;
        esac
        
        local payload=$(cat << EOF
{
    "attachments": [
        {
            "color": "$color",
            "title": "PostgreSQL Backup $status",
            "fields": [
                {
                    "title": "Database",
                    "value": "$DB_NAME ($DB_HOST)",
                    "short": true
                },
                {
                    "title": "Backup File",
                    "value": "$(basename "$backup_file")",
                    "short": true
                },
                {
                    "title": "Timestamp",
                    "value": "$(date '+%Y-%m-%d %H:%M:%S')",
                    "short": true
                }
            ]
        }
    ]
}
EOF
        )
        
        if curl -X POST -H 'Content-type: application/json' --data "$payload" "$SLACK_WEBHOOK_URL" >/dev/null 2>&1; then
            log INFO "Slack notification sent"
        else
            log WARN "Failed to send Slack notification"
        fi
    fi
}

# Cleanup function
cleanup() {
    local exit_code=$?
    
    # Clean up temporary files
    rm -f "$LOG_FILE.pg_dump" "/tmp/s3-lifecycle-config.json"
    
    # Clear password from environment
    unset PGPASSWORD
    
    if [[ $exit_code -eq 0 ]]; then
        log INFO "Backup process completed successfully"
        send_notification "SUCCESS" "${backup_path:-unknown}"
    else
        log ERROR "Backup process failed with exit code: $exit_code"
        send_notification "FAILED" "${backup_path:-unknown}"
    fi
    
    log INFO "Backup log saved to: $LOG_FILE"
    
    exit $exit_code
}

# Main backup function
main() {
    log INFO "Starting PostgreSQL backup process..."
    log INFO "Backup log: $LOG_FILE"
    
    # Parse command line arguments
    parse_args "$@"
    
    # Validate configuration
    validate_config
    
    # Test database connection
    test_connection
    
    # Get database size
    get_database_size
    
    # Perform backup
    local backup_path
    perform_backup
    
    # Clean up old backups
    cleanup_old_backups
    
    # Generate report
    generate_report "$backup_path"
    
    log INFO "Backup process completed successfully"
}

# Set trap for cleanup
trap cleanup EXIT

# Run main function with all arguments
main "$@"