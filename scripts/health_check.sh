#!/bin/bash
# System health monitoring script for PostgreSQL FDW Data Connector
set -euo pipefail

# Script configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
LOG_FILE="/tmp/health-check-$(date +%Y%m%d-%H%M%S).log"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
PURPLE='\033[0;35m'
NC='\033[0m' # No Color

# Default configuration
CHECK_INTERVAL="${CHECK_INTERVAL:-60}"
MAX_RETRIES="${MAX_RETRIES:-3}"
TIMEOUT="${TIMEOUT:-30}"
ENVIRONMENT="${ENVIRONMENT:-development}"
NAMESPACE="${NAMESPACE:-instant-connector}"
ALERT_WEBHOOK="${ALERT_WEBHOOK:-}"
ALERT_EMAIL="${ALERT_EMAIL:-}"
METRICS_ENABLED="${METRICS_ENABLED:-true}"
DETAILED_CHECKS="${DETAILED_CHECKS:-false}"

# Service endpoints
APP_ENDPOINT="${APP_ENDPOINT:-http://localhost:8000}"
DB_HOST="${DB_HOST:-localhost}"
DB_PORT="${DB_PORT:-5432}"
REDIS_HOST="${REDIS_HOST:-localhost}"
REDIS_PORT="${REDIS_PORT:-6379}"

# Health status
declare -A SERVICE_STATUS
declare -A SERVICE_DETAILS
declare -A SERVICE_METRICS

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
        OK)    echo -e "${GREEN}[OK]${NC} $message" | tee -a "$LOG_FILE" ;;
        FAIL)  echo -e "${RED}[FAIL]${NC} $message" | tee -a "$LOG_FILE" ;;
        *)     echo -e "${PURPLE}[$level]${NC} $message" | tee -a "$LOG_FILE" ;;
    esac
}

# Help function
show_help() {
    cat << EOF
System Health Monitoring Script

Usage: $(basename "$0") [OPTIONS]

OPTIONS:
    --environment ENV          Environment (development|staging|production) [default: development]
    --namespace NAMESPACE      Kubernetes namespace [default: instant-connector]
    --interval SECONDS         Check interval in seconds [default: 60]
    --timeout SECONDS          Request timeout [default: 30]
    --max-retries N            Maximum retries for failed checks [default: 3]
    --app-endpoint URL         Application endpoint [default: http://localhost:8000]
    --db-host HOST             Database host [default: localhost]
    --redis-host HOST          Redis host [default: localhost]
    --alert-webhook URL        Webhook URL for alerts
    --alert-email EMAIL        Email address for alerts
    --detailed                 Enable detailed system checks
    --once                     Run checks once and exit
    --daemon                   Run as daemon (continuous monitoring)
    --metrics                  Enable metrics collection
    --no-metrics               Disable metrics collection
    -h, --help                 Show this help message

EXAMPLES:
    # Run single health check
    $(basename "$0") --once

    # Run as daemon with 30-second intervals
    $(basename "$0") --daemon --interval 30

    # Production monitoring with alerts
    $(basename "$0") --environment production --alert-webhook https://hooks.slack.com/... --daemon

    # Kubernetes health checks
    $(basename "$0") --environment production --namespace instant-connector --detailed

ENVIRONMENT VARIABLES:
    CHECK_INTERVAL             Check interval in seconds
    MAX_RETRIES                Maximum retries for failed checks
    TIMEOUT                    Request timeout in seconds
    ENVIRONMENT                Deployment environment
    ALERT_WEBHOOK              Webhook URL for alerts
    ALERT_EMAIL                Email address for alerts
    METRICS_ENABLED            Enable metrics collection (true/false)

EOF
}

# Parse command line arguments
parse_args() {
    RUN_ONCE=false
    RUN_DAEMON=false
    
    while [[ $# -gt 0 ]]; do
        case $1 in
            --environment)
                ENVIRONMENT="$2"
                shift 2
                ;;
            --namespace)
                NAMESPACE="$2"
                shift 2
                ;;
            --interval)
                CHECK_INTERVAL="$2"
                shift 2
                ;;
            --timeout)
                TIMEOUT="$2"
                shift 2
                ;;
            --max-retries)
                MAX_RETRIES="$2"
                shift 2
                ;;
            --app-endpoint)
                APP_ENDPOINT="$2"
                shift 2
                ;;
            --db-host)
                DB_HOST="$2"
                shift 2
                ;;
            --redis-host)
                REDIS_HOST="$2"
                shift 2
                ;;
            --alert-webhook)
                ALERT_WEBHOOK="$2"
                shift 2
                ;;
            --alert-email)
                ALERT_EMAIL="$2"
                shift 2
                ;;
            --detailed)
                DETAILED_CHECKS="true"
                shift
                ;;
            --once)
                RUN_ONCE="true"
                shift
                ;;
            --daemon)
                RUN_DAEMON="true"
                shift
                ;;
            --metrics)
                METRICS_ENABLED="true"
                shift
                ;;
            --no-metrics)
                METRICS_ENABLED="false"
                shift
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

# Initialize health checking
initialize() {
    log INFO "Initializing health monitoring system..."
    log INFO "Environment: $ENVIRONMENT"
    log INFO "Check interval: ${CHECK_INTERVAL}s"
    log INFO "Max retries: $MAX_RETRIES"
    log INFO "Timeout: ${TIMEOUT}s"
    
    # Create metrics directory if metrics are enabled
    if [[ "$METRICS_ENABLED" == "true" ]]; then
        mkdir -p "/tmp/health-metrics"
    fi
}

# Check application health
check_application() {
    local service="application"
    local endpoint="$APP_ENDPOINT/health"
    
    log DEBUG "Checking application health: $endpoint"
    
    local start_time=$(date +%s.%N)
    local retries=0
    local success=false
    
    while [[ $retries -lt $MAX_RETRIES ]]; do
        if response=$(curl -sf --max-time "$TIMEOUT" "$endpoint" 2>/dev/null); then
            success=true
            break
        else
            ((retries++))
            if [[ $retries -lt $MAX_RETRIES ]]; then
                sleep 2
            fi
        fi
    done
    
    local end_time=$(date +%s.%N)
    local response_time=$(echo "$end_time - $start_time" | bc -l)
    
    if [[ "$success" == "true" ]]; then
        SERVICE_STATUS[$service]="healthy"
        SERVICE_DETAILS[$service]="Response time: ${response_time}s"
        SERVICE_METRICS[$service]=$response_time
        log OK "Application health check passed (${response_time}s)"
        
        # Parse JSON response if available
        if command -v jq >/dev/null && echo "$response" | jq . >/dev/null 2>&1; then
            local status=$(echo "$response" | jq -r '.status // "unknown"')
            local version=$(echo "$response" | jq -r '.version // "unknown"')
            SERVICE_DETAILS[$service]="Status: $status, Version: $version, Response: ${response_time}s"
        fi
    else
        SERVICE_STATUS[$service]="unhealthy"
        SERVICE_DETAILS[$service]="Failed after $retries retries"
        SERVICE_METRICS[$service]=0
        log FAIL "Application health check failed after $retries retries"
    fi
}

# Check database connectivity
check_database() {
    local service="database"
    
    log DEBUG "Checking database connectivity: $DB_HOST:$DB_PORT"
    
    local start_time=$(date +%s.%N)
    local success=false
    
    if command -v pg_isready >/dev/null; then
        if pg_isready -h "$DB_HOST" -p "$DB_PORT" -t "$TIMEOUT" >/dev/null 2>&1; then
            success=true
        fi
    else
        # Fallback to telnet/nc
        if command -v nc >/dev/null; then
            if nc -z -w "$TIMEOUT" "$DB_HOST" "$DB_PORT" 2>/dev/null; then
                success=true
            fi
        elif command -v telnet >/dev/null; then
            if timeout "$TIMEOUT" telnet "$DB_HOST" "$DB_PORT" </dev/null 2>/dev/null | grep -q "Connected"; then
                success=true
            fi
        fi
    fi
    
    local end_time=$(date +%s.%N)
    local response_time=$(echo "$end_time - $start_time" | bc -l)
    
    if [[ "$success" == "true" ]]; then
        SERVICE_STATUS[$service]="healthy"
        SERVICE_DETAILS[$service]="Connection time: ${response_time}s"
        SERVICE_METRICS[$service]=$response_time
        log OK "Database connectivity check passed (${response_time}s)"
    else
        SERVICE_STATUS[$service]="unhealthy"
        SERVICE_DETAILS[$service]="Connection failed"
        SERVICE_METRICS[$service]=0
        log FAIL "Database connectivity check failed"
    fi
}

# Check Redis connectivity
check_redis() {
    local service="redis"
    
    log DEBUG "Checking Redis connectivity: $REDIS_HOST:$REDIS_PORT"
    
    local start_time=$(date +%s.%N)
    local success=false
    
    if command -v redis-cli >/dev/null; then
        if redis-cli -h "$REDIS_HOST" -p "$REDIS_PORT" --connect-timeout "$TIMEOUT" ping >/dev/null 2>&1; then
            success=true
        fi
    else
        # Fallback to nc
        if command -v nc >/dev/null; then
            if nc -z -w "$TIMEOUT" "$REDIS_HOST" "$REDIS_PORT" 2>/dev/null; then
                success=true
            fi
        fi
    fi
    
    local end_time=$(date +%s.%N)
    local response_time=$(echo "$end_time - $start_time" | bc -l)
    
    if [[ "$success" == "true" ]]; then
        SERVICE_STATUS[$service]="healthy"
        SERVICE_DETAILS[$service]="Connection time: ${response_time}s"
        SERVICE_METRICS[$service]=$response_time
        log OK "Redis connectivity check passed (${response_time}s)"
    else
        SERVICE_STATUS[$service]="unhealthy"
        SERVICE_DETAILS[$service]="Connection failed"
        SERVICE_METRICS[$service]=0
        log FAIL "Redis connectivity check failed"
    fi
}

# Check system resources
check_system_resources() {
    local service="system"
    
    log DEBUG "Checking system resources..."
    
    local cpu_usage=""
    local memory_usage=""
    local disk_usage=""
    local load_average=""
    
    # CPU usage
    if command -v top >/dev/null; then
        cpu_usage=$(top -bn1 | grep "Cpu(s)" | awk '{print $2}' | sed 's/%us,//' || echo "unknown")
    fi
    
    # Memory usage
    if command -v free >/dev/null; then
        memory_usage=$(free | awk 'NR==2{printf "%.1f%%", $3*100/$2}' || echo "unknown")
    fi
    
    # Disk usage
    if command -v df >/dev/null; then
        disk_usage=$(df / | awk 'NR==2{print $5}' || echo "unknown")
    fi
    
    # Load average
    if [[ -f /proc/loadavg ]]; then
        load_average=$(cat /proc/loadavg | awk '{print $1, $2, $3}' || echo "unknown")
    fi
    
    # Determine health based on thresholds
    local cpu_threshold=80
    local memory_threshold=90
    local disk_threshold=85
    
    local status="healthy"
    local warnings=()
    
    if [[ "$cpu_usage" != "unknown" ]] && (( $(echo "$cpu_usage > $cpu_threshold" | bc -l) )); then
        warnings+=("High CPU usage: $cpu_usage")
        status="warning"
    fi
    
    if [[ "$memory_usage" != "unknown" ]] && (( $(echo "${memory_usage%\%} > $memory_threshold" | bc -l) )); then
        warnings+=("High memory usage: $memory_usage")
        status="warning"
    fi
    
    if [[ "$disk_usage" != "unknown" ]] && (( $(echo "${disk_usage%\%} > $disk_threshold" | bc -l) )); then
        warnings+=("High disk usage: $disk_usage")
        status="warning"
    fi
    
    SERVICE_STATUS[$service]=$status
    SERVICE_DETAILS[$service]="CPU: $cpu_usage, Memory: $memory_usage, Disk: $disk_usage, Load: $load_average"
    
    if [[ ${#warnings[@]} -gt 0 ]]; then
        log WARN "System resource warnings: ${warnings[*]}"
    else
        log OK "System resources check passed"
    fi
}

# Check Kubernetes pods (if applicable)
check_kubernetes_pods() {
    if [[ "$ENVIRONMENT" == "development" ]] || ! command -v kubectl >/dev/null; then
        return 0
    fi
    
    local service="kubernetes"
    
    log DEBUG "Checking Kubernetes pods in namespace: $NAMESPACE"
    
    local unhealthy_pods=()
    local total_pods=0
    local healthy_pods=0
    
    # Get pod status
    while IFS= read -r line; do
        if [[ -z "$line" ]]; then continue; fi
        
        local pod_name=$(echo "$line" | awk '{print $1}')
        local ready=$(echo "$line" | awk '{print $2}')
        local status=$(echo "$line" | awk '{print $3}')
        local restarts=$(echo "$line" | awk '{print $4}')
        
        ((total_pods++))
        
        if [[ "$status" == "Running" ]] && [[ "$ready" == *"/"* ]]; then
            local ready_count=$(echo "$ready" | cut -d'/' -f1)
            local total_count=$(echo "$ready" | cut -d'/' -f2)
            
            if [[ "$ready_count" == "$total_count" ]]; then
                ((healthy_pods++))
            else
                unhealthy_pods+=("$pod_name ($ready)")
            fi
        else
            unhealthy_pods+=("$pod_name ($status)")
        fi
    done < <(kubectl get pods -n "$NAMESPACE" --no-headers 2>/dev/null || echo "")
    
    if [[ $total_pods -eq 0 ]]; then
        SERVICE_STATUS[$service]="unknown"
        SERVICE_DETAILS[$service]="No pods found or kubectl access failed"
        log WARN "No Kubernetes pods found in namespace $NAMESPACE"
    elif [[ ${#unhealthy_pods[@]} -eq 0 ]]; then
        SERVICE_STATUS[$service]="healthy"
        SERVICE_DETAILS[$service]="All $total_pods pods healthy"
        log OK "All $total_pods Kubernetes pods are healthy"
    else
        SERVICE_STATUS[$service]="unhealthy"
        SERVICE_DETAILS[$service]="$healthy_pods/$total_pods pods healthy. Unhealthy: ${unhealthy_pods[*]}"
        log FAIL "Unhealthy Kubernetes pods: ${unhealthy_pods[*]}"
    fi
}

# Check detailed application metrics
check_detailed_metrics() {
    if [[ "$DETAILED_CHECKS" != "true" ]]; then
        return 0
    fi
    
    log DEBUG "Running detailed application metrics checks..."
    
    # Check FDW connections
    local fdw_endpoint="$APP_ENDPOINT/api/health/fdw"
    if response=$(curl -sf --max-time "$TIMEOUT" "$fdw_endpoint" 2>/dev/null); then
        log OK "FDW health check passed"
        if command -v jq >/dev/null; then
            local fdw_status=$(echo "$response" | jq -r '.status // "unknown"')
            local connection_count=$(echo "$response" | jq -r '.active_connections // 0')
            SERVICE_DETAILS["fdw"]="Status: $fdw_status, Connections: $connection_count"
        fi
    else
        log WARN "FDW health check failed or not available"
    fi
    
    # Check Celery workers
    local celery_endpoint="$APP_ENDPOINT/api/health/celery"
    if response=$(curl -sf --max-time "$TIMEOUT" "$celery_endpoint" 2>/dev/null); then
        log OK "Celery health check passed"
        if command -v jq >/dev/null; then
            local worker_count=$(echo "$response" | jq -r '.active_workers // 0')
            local queue_size=$(echo "$response" | jq -r '.queue_size // 0')
            SERVICE_DETAILS["celery"]="Workers: $worker_count, Queue: $queue_size"
        fi
    else
        log WARN "Celery health check failed or not available"
    fi
}

# Collect metrics
collect_metrics() {
    if [[ "$METRICS_ENABLED" != "true" ]]; then
        return 0
    fi
    
    local metrics_file="/tmp/health-metrics/health_metrics_$(date +%Y%m%d_%H%M%S).json"
    local timestamp=$(date +%s)
    
    # Build metrics JSON
    local metrics_json="{"
    metrics_json+="\"timestamp\": $timestamp,"
    metrics_json+="\"environment\": \"$ENVIRONMENT\","
    metrics_json+="\"services\": {"
    
    local first=true
    for service in "${!SERVICE_STATUS[@]}"; do
        if [[ "$first" == "false" ]]; then
            metrics_json+=","
        fi
        first=false
        
        local status="${SERVICE_STATUS[$service]}"
        local details="${SERVICE_DETAILS[$service]:-}"
        local response_time="${SERVICE_METRICS[$service]:-0}"
        
        metrics_json+="\"$service\": {"
        metrics_json+="\"status\": \"$status\","
        metrics_json+="\"details\": \"$details\","
        metrics_json+="\"response_time\": $response_time"
        metrics_json+="}"
    done
    
    metrics_json+="}}"
    
    # Save metrics to file
    echo "$metrics_json" > "$metrics_file"
    
    # Send to metrics collector if configured
    if [[ -n "${METRICS_ENDPOINT:-}" ]]; then
        curl -X POST -H "Content-Type: application/json" -d "$metrics_json" "$METRICS_ENDPOINT" >/dev/null 2>&1 || true
    fi
    
    log DEBUG "Metrics collected: $metrics_file"
}

# Generate health report
generate_health_report() {
    log INFO "=== HEALTH CHECK REPORT ==="
    log INFO "Timestamp: $(date '+%Y-%m-%d %H:%M:%S')"
    log INFO "Environment: $ENVIRONMENT"
    
    local overall_status="healthy"
    local unhealthy_services=()
    local warning_services=()
    
    for service in "${!SERVICE_STATUS[@]}"; do
        local status="${SERVICE_STATUS[$service]}"
        local details="${SERVICE_DETAILS[$service]:-}"
        
        case "$status" in
            "healthy")
                log INFO "✓ $service: HEALTHY - $details"
                ;;
            "warning")
                log WARN "⚠ $service: WARNING - $details"
                warning_services+=("$service")
                if [[ "$overall_status" == "healthy" ]]; then
                    overall_status="warning"
                fi
                ;;
            "unhealthy")
                log FAIL "✗ $service: UNHEALTHY - $details"
                unhealthy_services+=("$service")
                overall_status="unhealthy"
                ;;
            *)
                log WARN "? $service: UNKNOWN - $details"
                warning_services+=("$service")
                if [[ "$overall_status" == "healthy" ]]; then
                    overall_status="warning"
                fi
                ;;
        esac
    done
    
    log INFO "Overall Status: $(echo "$overall_status" | tr '[:lower:]' '[:upper:]')"
    
    if [[ ${#unhealthy_services[@]} -gt 0 ]]; then
        log ERROR "Unhealthy services: ${unhealthy_services[*]}"
    fi
    
    if [[ ${#warning_services[@]} -gt 0 ]]; then
        log WARN "Services with warnings: ${warning_services[*]}"
    fi
    
    log INFO "=========================="
    
    # Send alerts if needed
    if [[ "$overall_status" != "healthy" ]]; then
        send_alert "$overall_status" "${unhealthy_services[*]} ${warning_services[*]}"
    fi
    
    return $(if [[ "$overall_status" == "healthy" ]]; then echo 0; else echo 1; fi)
}

# Send alert notifications
send_alert() {
    local status="$1"
    local services="$2"
    
    local alert_message="Health Check Alert - $ENVIRONMENT
Status: $(echo "$status" | tr '[:lower:]' '[:upper:]')
Affected services: $services
Timestamp: $(date '+%Y-%m-%d %H:%M:%S')
Environment: $ENVIRONMENT"
    
    # Webhook notification
    if [[ -n "$ALERT_WEBHOOK" ]]; then
        local color
        case "$status" in
            "unhealthy") color="danger" ;;
            "warning") color="warning" ;;
            *) color="good" ;;
        esac
        
        local webhook_payload="{
            \"text\": \"Health Check Alert\",
            \"attachments\": [{
                \"color\": \"$color\",
                \"title\": \"System Health Alert - $ENVIRONMENT\",
                \"text\": \"$alert_message\",
                \"ts\": $(date +%s)
            }]
        }"
        
        if curl -X POST -H 'Content-type: application/json' --data "$webhook_payload" "$ALERT_WEBHOOK" >/dev/null 2>&1; then
            log INFO "Alert sent to webhook"
        else
            log WARN "Failed to send webhook alert"
        fi
    fi
    
    # Email notification
    if [[ -n "$ALERT_EMAIL" ]] && command -v mail >/dev/null; then
        local subject="Health Check Alert - $ENVIRONMENT - $(echo "$status" | tr '[:lower:]' '[:upper:]')"
        
        if echo "$alert_message" | mail -s "$subject" "$ALERT_EMAIL"; then
            log INFO "Alert sent to email: $ALERT_EMAIL"
        else
            log WARN "Failed to send email alert"
        fi
    fi
}

# Run single health check
run_health_check() {
    log INFO "Running health checks..."
    
    # Clear previous status
    SERVICE_STATUS=()
    SERVICE_DETAILS=()
    SERVICE_METRICS=()
    
    # Run all health checks
    check_application
    check_database
    check_redis
    check_system_resources
    check_kubernetes_pods
    check_detailed_metrics
    
    # Collect metrics
    collect_metrics
    
    # Generate report
    generate_health_report
}

# Run continuous monitoring
run_daemon() {
    log INFO "Starting health monitoring daemon..."
    log INFO "Check interval: ${CHECK_INTERVAL} seconds"
    log INFO "Press Ctrl+C to stop"
    
    # Set up signal handlers
    trap 'log INFO "Shutting down health monitoring daemon..."; exit 0' SIGTERM SIGINT
    
    while true; do
        run_health_check
        
        log INFO "Next check in ${CHECK_INTERVAL} seconds..."
        sleep "$CHECK_INTERVAL"
    done
}

# Main function
main() {
    log INFO "Starting health monitoring system..."
    
    # Parse command line arguments
    parse_args "$@"
    
    # Initialize
    initialize
    
    # Run based on mode
    if [[ "$RUN_DAEMON" == "true" ]]; then
        run_daemon
    else
        run_health_check
        
        if [[ "$RUN_ONCE" == "true" ]]; then
            log INFO "Single health check completed"
        fi
    fi
}

# Run main function with all arguments
main "$@"