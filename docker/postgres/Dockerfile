# PostgreSQL with FDW Extensions
FROM postgres:15-alpine

# Set build arguments
ARG POSTGRES_VERSION=15
ARG BUILD_DATE
ARG VERSION

# Add metadata labels
LABEL maintainer="Data Engineering Team" \
      org.label-schema.build-date=$BUILD_DATE \
      org.label-schema.name="postgres-fdw-extended" \
      org.label-schema.description="PostgreSQL with FDW extensions pre-installed" \
      org.label-schema.version=$VERSION \
      org.label-schema.schema-version="1.0"

# Install build dependencies
RUN apk add --no-cache --virtual .build-deps \
    gcc \
    musl-dev \
    make \
    postgresql-dev \
    python3-dev \
    py3-pip \
    git \
    curl \
    tar \
    gzip

# Install runtime dependencies
RUN apk add --no-cache \
    python3 \
    py3-psycopg2 \
    mysql-client \
    curl \
    ca-certificates

# Install Python packages for Multicorn FDW
RUN pip3 install --no-cache-dir \
    multicorn \
    requests \
    pymysql \
    psycopg2-binary \
    sqlalchemy \
    pandas

# Create extension build directory
WORKDIR /tmp/extensions

# Install postgres_fdw (usually included in contrib)
RUN echo "postgres_fdw is included in PostgreSQL contrib package"

# Install file_fdw (usually included in contrib)
RUN echo "file_fdw is included in PostgreSQL contrib package"

# Install mysql_fdw
RUN git clone https://github.com/EnterpriseDB/mysql_fdw.git && \
    cd mysql_fdw && \
    make USE_PGXS=1 && \
    make USE_PGXS=1 install

# Install multicorn (Python FDW framework)
# Note: Multicorn installation via pip should be sufficient for basic usage

# Install additional useful extensions
RUN apk add --no-cache postgresql-contrib

# Clean up build dependencies
RUN apk del .build-deps && \
    rm -rf /tmp/extensions /var/cache/apk/*

# Copy initialization scripts
COPY docker/postgres/init-scripts/ /docker-entrypoint-initdb.d/
COPY docker/postgres/configs/ /etc/postgresql/

# Copy custom configuration
COPY docker/postgres/postgresql.conf /tmp/postgresql.conf
COPY docker/postgres/pg_hba.conf /tmp/pg_hba.conf

# Set proper permissions
RUN chmod 644 /tmp/postgresql.conf /tmp/pg_hba.conf

# Environment variables for PostgreSQL
ENV POSTGRES_DB=instant_connector
ENV POSTGRES_USER=connector_user
ENV POSTGRES_PASSWORD=secure_password_change_me
ENV POSTGRES_INITDB_ARGS="--auth-host=scram-sha-256 --auth-local=scram-sha-256"

# Expose PostgreSQL port
EXPOSE 5432

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD pg_isready -U $POSTGRES_USER -d $POSTGRES_DB || exit 1

# Use custom entrypoint for additional setup
COPY docker/postgres/docker-entrypoint.sh /usr/local/bin/
RUN chmod +x /usr/local/bin/docker-entrypoint.sh

ENTRYPOINT ["/usr/local/bin/docker-entrypoint.sh"]
CMD ["postgres"]