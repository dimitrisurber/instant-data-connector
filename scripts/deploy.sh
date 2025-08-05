#!/bin/bash
# Deployment automation script for PostgreSQL FDW Data Connector
set -euo pipefail

# Script configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
LOG_FILE="/tmp/deploy-$(date +%Y%m%d-%H%M%S).log"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
PURPLE='\033[0;35m'
NC='\033[0m' # No Color

# Default values
ENVIRONMENT="development"
DOCKER_REGISTRY=""
IMAGE_TAG=""
NAMESPACE="instant-connector"
DRY_RUN=false
SKIP_BUILD=false
SKIP_TESTS=false
FORCE_RECREATE=false
BACKUP_BEFORE_DEPLOY=true

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
        *)     echo -e "${PURPLE}[$level]${NC} $message" | tee -a "$LOG_FILE" ;;
    esac
}

# Help function
show_help() {
    cat << EOF
PostgreSQL FDW Data Connector Deployment Script

Usage: $(basename "$0") [OPTIONS]

OPTIONS:
    -e, --environment ENV       Deployment environment (development|staging|production) [default: development]
    -r, --registry REGISTRY     Docker registry URL
    -t, --tag TAG              Docker image tag [default: latest]
    -n, --namespace NAMESPACE  Kubernetes namespace [default: instant-connector]
    --dry-run                  Show what would be deployed without actually deploying
    --skip-build               Skip Docker image build
    --skip-tests               Skip running tests
    --force-recreate           Force recreate all resources
    --no-backup                Skip backup before deployment
    -h, --help                 Show this help message

EXAMPLES:
    # Deploy to development environment
    $(basename "$0") -e development

    # Deploy to production with custom image tag
    $(basename "$0") -e production -t v1.2.3 -r registry.example.com

    # Dry run for production deployment
    $(basename "$0") -e production --dry-run

    # Deploy without running tests (not recommended for production)
    $(basename "$0") -e staging --skip-tests

ENVIRONMENT VARIABLES:
    DOCKER_REGISTRY         Default Docker registry
    KUBECONFIG             Kubernetes configuration file
    POSTGRES_PASSWORD      PostgreSQL password
    REDIS_PASSWORD         Redis password
    SECRET_KEY             Application secret key

EOF
}

# Parse command line arguments
parse_args() {
    while [[ $# -gt 0 ]]; do
        case $1 in
            -e|--environment)
                ENVIRONMENT="$2"
                shift 2
                ;;
            -r|--registry)
                DOCKER_REGISTRY="$2"
                shift 2
                ;;
            -t|--tag)
                IMAGE_TAG="$2"
                shift 2
                ;;
            -n|--namespace)
                NAMESPACE="$2"
                shift 2
                ;;
            --dry-run)
                DRY_RUN=true
                shift
                ;;
            --skip-build)
                SKIP_BUILD=true
                shift
                ;;
            --skip-tests)
                SKIP_TESTS=true
                shift
                ;;
            --force-recreate)
                FORCE_RECREATE=true
                shift
                ;;
            --no-backup)
                BACKUP_BEFORE_DEPLOY=false
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

# Validate environment
validate_environment() {
    log INFO "Validating deployment environment..."
    
    case "$ENVIRONMENT" in
        development|staging|production)
            log INFO "Environment: $ENVIRONMENT"
            ;;
        *)
            log ERROR "Invalid environment: $ENVIRONMENT"
            log ERROR "Valid environments: development, staging, production"
            exit 1
            ;;
    esac
    
    # Set default image tag if not provided
    if [[ -z "$IMAGE_TAG" ]]; then
        if [[ "$ENVIRONMENT" == "production" ]]; then
            IMAGE_TAG="$(git describe --tags --always 2>/dev/null || echo 'latest')"
        else
            IMAGE_TAG="latest"
        fi
    fi
    
    log INFO "Image tag: $IMAGE_TAG"
    log INFO "Namespace: $NAMESPACE"
    log INFO "Docker registry: ${DOCKER_REGISTRY:-'default'}"
}

# Check prerequisites
check_prerequisites() {
    log INFO "Checking prerequisites..."
    
    local missing_tools=()
    
    # Check required tools
    command -v docker >/dev/null || missing_tools+=("docker")
    command -v kubectl >/dev/null || missing_tools+=("kubectl")
    command -v git >/dev/null || missing_tools+=("git")
    
    if [[ "$ENVIRONMENT" == "development" ]]; then
        command -v docker-compose >/dev/null || missing_tools+=("docker-compose")
    fi
    
    if [[ ${#missing_tools[@]} -gt 0 ]]; then
        log ERROR "Missing required tools: ${missing_tools[*]}"
        log ERROR "Please install the missing tools and try again"
        exit 1
    fi
    
    # Check Docker is running
    if ! docker info >/dev/null 2>&1; then
        log ERROR "Docker is not running. Please start Docker and try again"
        exit 1
    fi
    
    # Check Kubernetes connection (for k8s deployments)
    if [[ "$ENVIRONMENT" != "development" ]]; then
        if ! kubectl cluster-info >/dev/null 2>&1; then
            log ERROR "Cannot connect to Kubernetes cluster"
            log ERROR "Please check your KUBECONFIG and try again"
            exit 1
        fi
        
        log INFO "Kubernetes cluster: $(kubectl config current-context)"
    fi
    
    log INFO "Prerequisites check passed"
}

# Run tests
run_tests() {
    if [[ "$SKIP_TESTS" == true ]]; then
        log WARN "Skipping tests (--skip-tests flag provided)"
        return 0
    fi
    
    log INFO "Running tests..."
    
    cd "$PROJECT_ROOT"
    
    # Create test environment if it doesn't exist
    if [[ ! -f "test_env/bin/activate" ]]; then
        log INFO "Creating test virtual environment..."
        python3 -m venv test_env
        source test_env/bin/activate
        pip install -r requirements.txt -r requirements-dev.txt
    else
        source test_env/bin/activate
    fi
    
    # Run tests
    if ! python -m pytest tests/ -v --tb=short; then
        log ERROR "Tests failed"
        if [[ "$ENVIRONMENT" == "production" ]]; then
            log ERROR "Cannot deploy to production with failing tests"
            exit 1
        else
            log WARN "Tests failed but continuing with deployment (non-production environment)"
        fi
    else
        log INFO "All tests passed"
    fi
    
    deactivate
}

# Build Docker images
build_images() {
    if [[ "$SKIP_BUILD" == true ]]; then
        log WARN "Skipping Docker image build (--skip-build flag provided)"
        return 0
    fi
    
    log INFO "Building Docker images..."
    
    cd "$PROJECT_ROOT"
    
    local build_args=(
        --build-arg "BUILD_DATE=$(date -u +'%Y-%m-%dT%H:%M:%SZ')"
        --build-arg "VERSION=$IMAGE_TAG"
        --build-arg "VCS_REF=$(git rev-parse --short HEAD 2>/dev/null || echo 'unknown')"
    )
    
    # Build main application image
    local app_image="instant-data-connector:$IMAGE_TAG"
    if [[ -n "$DOCKER_REGISTRY" ]]; then
        app_image="$DOCKER_REGISTRY/instant-data-connector:$IMAGE_TAG"
    fi
    
    log INFO "Building application image: $app_image"
    if [[ "$DRY_RUN" == false ]]; then
        docker build "${build_args[@]}" -t "$app_image" .
    fi
    
    # Build PostgreSQL FDW image
    local postgres_image="instant-data-connector/postgres-fdw:$IMAGE_TAG"
    if [[ -n "$DOCKER_REGISTRY" ]]; then
        postgres_image="$DOCKER_REGISTRY/postgres-fdw:$IMAGE_TAG"
    fi
    
    log INFO "Building PostgreSQL FDW image: $postgres_image"
    if [[ "$DRY_RUN" == false ]]; then
        docker build "${build_args[@]}" -t "$postgres_image" -f docker/postgres/Dockerfile .
    fi
    
    # Push images to registry if specified
    if [[ -n "$DOCKER_REGISTRY" && "$DRY_RUN" == false ]]; then
        log INFO "Pushing images to registry..."
        docker push "$app_image"
        docker push "$postgres_image"
    fi
    
    log INFO "Docker images built successfully"
}

# Deploy to development environment
deploy_development() {
    log INFO "Deploying to development environment..."
    
    cd "$PROJECT_ROOT"
    
    # Create .env file for development
    cat > ".env" << EOF
BUILD_DATE=$(date -u +'%Y-%m-%dT%H:%M:%SZ')
VERSION=$IMAGE_TAG
VCS_REF=$(git rev-parse --short HEAD 2>/dev/null || echo 'unknown')

# Database
POSTGRES_DB=instant_connector
POSTGRES_USER=connector_user
POSTGRES_PASSWORD=${POSTGRES_PASSWORD:-dev_password_123}

# Redis
REDIS_PASSWORD=${REDIS_PASSWORD:-redis_password_123}

# Application
SECRET_KEY=${SECRET_KEY:-super_secret_key_change_me_in_production}
ENCRYPTION_KEY=${ENCRYPTION_KEY:-encryption_key_change_me_in_production}
JWT_SECRET=${JWT_SECRET:-jwt_secret_change_me_in_production}
LOG_LEVEL=DEBUG
EOF
    
    if [[ "$DRY_RUN" == true ]]; then
        log INFO "DRY RUN: Would execute: docker-compose up -d"
        return 0
    fi
    
    # Stop existing containers if force recreate
    if [[ "$FORCE_RECREATE" == true ]]; then
        log INFO "Force recreating containers..."
        docker-compose down -v
    fi
    
    # Start services
    docker-compose up -d
    
    # Wait for services to be ready
    log INFO "Waiting for services to be ready..."
    sleep 30
    
    # Check service health
    check_service_health_development
    
    log INFO "Development deployment completed successfully"
    log INFO "Services are available at:"
    log INFO "  - Application: http://localhost:8000"
    log INFO "  - Database Admin: http://localhost:8080"
    log INFO "  - Redis Admin: http://localhost:8081"
    log INFO "  - Grafana: http://localhost:3000"
    log INFO "  - Prometheus: http://localhost:9090"
}

# Deploy to Kubernetes
deploy_kubernetes() {
    log INFO "Deploying to Kubernetes environment: $ENVIRONMENT"
    
    cd "$PROJECT_ROOT"
    
    # Backup before deployment if requested
    if [[ "$BACKUP_BEFORE_DEPLOY" == true && "$ENVIRONMENT" == "production" ]]; then
        backup_database
    fi
    
    # Create namespace if it doesn't exist
    if [[ "$DRY_RUN" == false ]]; then
        kubectl create namespace "$NAMESPACE" --dry-run=client -o yaml | kubectl apply -f -
    else
        log INFO "DRY RUN: Would create namespace: $NAMESPACE"
    fi
    
    # Apply Kubernetes manifests
    local manifests=(
        "deployment/kubernetes/namespace.yaml"
        "deployment/kubernetes/configmap.yaml"
        "deployment/kubernetes/secret.yaml"
        "deployment/kubernetes/persistentvolume.yaml"
        "deployment/kubernetes/deployment.yaml"
        "deployment/kubernetes/service.yaml"
        "deployment/kubernetes/ingress.yaml"
    )
    
    for manifest in "${manifests[@]}"; do
        if [[ -f "$manifest" ]]; then
            log INFO "Applying manifest: $manifest"
            if [[ "$DRY_RUN" == false ]]; then
                # Replace placeholders in manifests
                sed -e "s|instant-data-connector:latest|$app_image|g" \
                    -e "s|instant-data-connector/postgres-fdw:latest|$postgres_image|g" \
                    -e "s|namespace: instant-connector|namespace: $NAMESPACE|g" \
                    "$manifest" | kubectl apply -f - -n "$NAMESPACE"
            else
                log INFO "DRY RUN: Would apply manifest: $manifest"
            fi
        else
            log WARN "Manifest not found: $manifest"
        fi
    done
    
    if [[ "$DRY_RUN" == false ]]; then
        # Wait for deployment to complete
        log INFO "Waiting for deployment to complete..."
        kubectl rollout status deployment/connector-app -n "$NAMESPACE" --timeout=600s
        kubectl rollout status deployment/postgres-fdw -n "$NAMESPACE" --timeout=600s
        kubectl rollout status deployment/redis -n "$NAMESPACE" --timeout=300s
        kubectl rollout status deployment/nginx -n "$NAMESPACE" --timeout=300s
        
        # Check service health
        check_service_health_kubernetes
        
        log INFO "Kubernetes deployment completed successfully"
        
        # Show service endpoints
        show_service_endpoints
    else
        log INFO "DRY RUN: Kubernetes deployment simulation completed"
    fi
}

# Backup database
backup_database() {
    log INFO "Creating database backup before deployment..."
    
    if [[ "$DRY_RUN" == true ]]; then
        log INFO "DRY RUN: Would create database backup"
        return 0
    fi
    
    # Run backup script
    if [[ -f "$SCRIPT_DIR/backup.sh" ]]; then
        bash "$SCRIPT_DIR/backup.sh" --auto-backup
    else
        log WARN "Backup script not found, skipping backup"
    fi
}

# Check service health for development
check_service_health_development() {
    log INFO "Checking service health..."
    
    local services=(
        "http://localhost:8000/health:Application"
        "http://localhost:8080:Database Admin"
        "http://localhost:6379:Redis"
    )
    
    for service in "${services[@]}"; do
        local url="${service%:*}"
        local name="${service#*:}"
        
        log INFO "Checking $name health..."
        
        local retries=0
        local max_retries=10
        
        while [[ $retries -lt $max_retries ]]; do
            if curl -sf "$url" >/dev/null 2>&1; then
                log INFO "$name is healthy"
                break
            else
                ((retries++))
                if [[ $retries -eq $max_retries ]]; then
                    log ERROR "$name health check failed after $max_retries attempts"
                else
                    log INFO "$name not ready, retrying in 10 seconds... ($retries/$max_retries)"
                    sleep 10
                fi
            fi
        done
    done
}

# Check service health for Kubernetes
check_service_health_kubernetes() {
    log INFO "Checking Kubernetes service health..."
    
    # Check pod status
    log INFO "Pod status:"
    kubectl get pods -n "$NAMESPACE" -o wide
    
    # Check service endpoints
    log INFO "Service endpoints:"
    kubectl get svc -n "$NAMESPACE"
    
    # Check application health endpoint
    local app_pod=$(kubectl get pods -n "$NAMESPACE" -l app=instant-data-connector -o jsonpath='{.items[0].metadata.name}' 2>/dev/null || echo "")
    
    if [[ -n "$app_pod" ]]; then
        log INFO "Checking application health endpoint..."
        if kubectl exec -n "$NAMESPACE" "$app_pod" -- curl -sf http://localhost:8000/health >/dev/null 2>&1; then
            log INFO "Application health check passed"
        else
            log WARN "Application health check failed"
        fi
    fi
}

# Show service endpoints
show_service_endpoints() {
    log INFO "Service endpoints:"
    
    # Get ingress information
    local ingress_info=$(kubectl get ingress -n "$NAMESPACE" -o jsonpath='{.items[*].spec.rules[*].host}' 2>/dev/null || echo "")
    
    if [[ -n "$ingress_info" ]]; then
        log INFO "Ingress endpoints:"
        for host in $ingress_info; do
            log INFO "  - https://$host"
        done
    fi
    
    # Get LoadBalancer services
    local lb_services=$(kubectl get svc -n "$NAMESPACE" -o jsonpath='{range .items[?(@.spec.type=="LoadBalancer")]}{.metadata.name}{"\t"}{.status.loadBalancer.ingress[0].ip}{"\t"}{.spec.ports[0].port}{"\n"}{end}' 2>/dev/null || echo "")
    
    if [[ -n "$lb_services" ]]; then
        log INFO "LoadBalancer services:"
        echo "$lb_services" | while IFS=$'\t' read -r name ip port; do
            if [[ -n "$ip" ]]; then
                log INFO "  - $name: http://$ip:$port"
            fi
        done
    fi
}

# Cleanup function
cleanup() {
    local exit_code=$?
    
    if [[ $exit_code -eq 0 ]]; then
        log INFO "Deployment completed successfully"
    else
        log ERROR "Deployment failed with exit code: $exit_code"
    fi
    
    log INFO "Deployment log saved to: $LOG_FILE"
    
    exit $exit_code
}

# Main deployment function
main() {
    log INFO "Starting deployment process..."
    log INFO "Deployment log: $LOG_FILE"
    
    # Parse command line arguments
    parse_args "$@"
    
    # Validate environment
    validate_environment
    
    # Check prerequisites
    check_prerequisites
    
    # Run tests
    run_tests
    
    # Build Docker images
    build_images
    
    # Deploy based on environment
    case "$ENVIRONMENT" in
        development)
            deploy_development
            ;;
        staging|production)
            deploy_kubernetes
            ;;
    esac
    
    log INFO "Deployment process completed"
}

# Set trap for cleanup
trap cleanup EXIT

# Run main function with all arguments
main "$@"