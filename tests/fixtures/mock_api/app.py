"""
Mock REST API server for testing FDW integrations.

This FastAPI application provides mock endpoints that simulate real-world APIs
for testing the FDW-based data connector integration patterns.
"""

import os
import asyncio
import random
from datetime import datetime, timedelta
from typing import Optional, List, Dict, Any
from faker import Faker
from fastapi import FastAPI, HTTPException, Query, Path
from fastapi.responses import JSONResponse
import uvicorn

# Initialize Faker
fake = Faker()

# Application configuration
API_HOST = os.getenv("API_HOST", "0.0.0.0")
API_PORT = int(os.getenv("API_PORT", "8000"))
LOG_LEVEL = os.getenv("LOG_LEVEL", "info")

# Initialize FastAPI app
app = FastAPI(
    title="Mock API Server for FDW Testing",
    description="Provides mock data endpoints for testing FDW integrations",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# Global data stores (in-memory for testing)
users_db = []
orders_db = []
products_db = []
analytics_db = []

def generate_test_data():
    """Generate initial test data."""
    global users_db, orders_db, products_db, analytics_db
    
    # Generate users
    users_db = []
    for i in range(1000):
        user = {
            "user_id": i + 1,
            "username": fake.user_name(),
            "first_name": fake.first_name(),
            "last_name": fake.last_name(),
            "email": fake.email(),
            "registration_date": fake.date_between(start_date='-2y', end_date='today').isoformat(),
            "last_login": fake.date_time_between(start_date='-30d', end_date='now').isoformat() if random.random() > 0.1 else None,
            "is_active": random.random() > 0.15,
            "profile": {
                "age": random.randint(18, 80),
                "country": fake.country_code(),
                "city": fake.city(),
                "timezone": fake.timezone(),
                "preferences": {
                    "newsletter": random.random() > 0.3,
                    "notifications": random.random() > 0.4,
                    "theme": random.choice(["light", "dark", "auto"])
                }
            },
            "created_at": fake.date_time_between(start_date='-2y', end_date='-1y').isoformat(),
            "updated_at": fake.date_time_between(start_date='-1y', end_date='now').isoformat()
        }
        users_db.append(user)
    
    # Generate products
    categories = ["Electronics", "Books", "Clothing", "Home & Garden", "Sports", "Toys", "Health", "Automotive"]
    products_db = []
    for i in range(500):
        product = {
            "product_id": i + 1,
            "sku": f"PROD-{i+1:04d}",
            "name": fake.catch_phrase(),
            "description": fake.text(max_nb_chars=200),
            "category": random.choice(categories),
            "price": round(random.uniform(10.0, 500.0), 2),
            "cost": round(random.uniform(5.0, 300.0), 2),
            "weight": round(random.uniform(0.1, 10.0), 3),
            "dimensions": {
                "length": round(random.uniform(5.0, 50.0), 1),
                "width": round(random.uniform(5.0, 50.0), 1),
                "height": round(random.uniform(2.0, 20.0), 1)
            },
            "stock_quantity": random.randint(0, 1000),
            "is_active": random.random() > 0.1,
            "rating": round(random.uniform(1.0, 5.0), 1),
            "review_count": random.randint(0, 500),
            "created_at": fake.date_time_between(start_date='-1y', end_date='-6m').isoformat(),
            "updated_at": fake.date_time_between(start_date='-6m', end_date='now').isoformat()
        }
        products_db.append(product)
    
    # Generate orders
    statuses = ["pending", "processing", "shipped", "delivered", "cancelled", "returned"]
    orders_db = []
    for i in range(2000):
        customer_id = random.randint(1, 1000)
        order_date = fake.date_time_between(start_date='-1y', end_date='now')
        
        order = {
            "order_id": i + 1,
            "order_number": f"ORD-{fake.year()}-{i+1:06d}",
            "customer_id": customer_id,
            "status": random.choice(statuses),
            "total_amount": round(random.uniform(25.0, 1000.0), 2),
            "tax_amount": round(random.uniform(2.0, 80.0), 2),
            "shipping_amount": round(random.uniform(0.0, 25.0), 2),
            "discount_amount": round(random.uniform(0.0, 50.0), 2),
            "order_date": order_date.isoformat(),
            "shipped_date": (order_date + timedelta(days=random.randint(1, 7))).isoformat() if random.random() > 0.3 else None,
            "delivered_date": (order_date + timedelta(days=random.randint(3, 14))).isoformat() if random.random() > 0.4 else None,
            "items_count": random.randint(1, 8),
            "shipping_address": {
                "street": fake.street_address(),
                "city": fake.city(),
                "state": fake.state_abbr(),
                "zip_code": fake.zipcode(),
                "country": "US"
            },
            "payment_method": random.choice(["credit_card", "debit_card", "paypal", "apple_pay", "google_pay"]),
            "created_at": order_date.isoformat(),
            "updated_at": fake.date_time_between(start_date=order_date, end_date='now').isoformat()
        }
        orders_db.append(order)
    
    # Generate analytics data
    analytics_db = []
    base_date = datetime.now() - timedelta(days=365)
    for i in range(365):
        date = base_date + timedelta(days=i)
        analytics = {
            "date": date.date().isoformat(),
            "page_views": random.randint(1000, 10000),
            "unique_visitors": random.randint(500, 5000),
            "bounce_rate": round(random.uniform(0.2, 0.8), 3),
            "avg_session_duration": random.randint(120, 1800),  # seconds
            "conversion_rate": round(random.uniform(0.01, 0.05), 4),
            "revenue": round(random.uniform(1000.0, 50000.0), 2),
            "orders_count": random.randint(10, 200),
            "new_users": random.randint(5, 100),
            "returning_users": random.randint(20, 400),
            "traffic_sources": {
                "organic": round(random.uniform(0.3, 0.6), 3),
                "paid": round(random.uniform(0.1, 0.3), 3),
                "social": round(random.uniform(0.05, 0.2), 3),
                "direct": round(random.uniform(0.1, 0.3), 3),
                "email": round(random.uniform(0.05, 0.15), 3)
            }
        }
        analytics_db.append(analytics)

# Generate initial data
generate_test_data()

@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "version": "1.0.0",
        "data_counts": {
            "users": len(users_db),
            "orders": len(orders_db),
            "products": len(products_db),
            "analytics": len(analytics_db)
        }
    }

@app.get("/")
async def root():
    """Root endpoint with API information."""
    return {
        "message": "Mock API Server for FDW Testing",
        "version": "1.0.0",
        "endpoints": {
            "users": "/users",
            "orders": "/orders",
            "products": "/products",
            "analytics": "/analytics",
            "health": "/health",
            "docs": "/docs"
        }
    }

# Users endpoints
@app.get("/users")
async def get_users(
    limit: int = Query(default=100, ge=1, le=1000),
    offset: int = Query(default=0, ge=0),
    active_only: bool = Query(default=False),
    country: Optional[str] = Query(default=None),
    registered_after: Optional[str] = Query(default=None)
):
    """Get users with filtering and pagination."""
    filtered_users = users_db.copy()
    
    # Apply filters
    if active_only:
        filtered_users = [u for u in filtered_users if u["is_active"]]
    
    if country:
        filtered_users = [u for u in filtered_users if u["profile"]["country"] == country]
    
    if registered_after:
        filtered_users = [u for u in filtered_users if u["registration_date"] >= registered_after]
    
    # Apply pagination
    total = len(filtered_users)
    paginated_users = filtered_users[offset:offset + limit]
    
    return {
        "data": paginated_users,
        "pagination": {
            "total": total,
            "limit": limit,
            "offset": offset,
            "has_next": offset + limit < total,
            "has_previous": offset > 0
        }
    }

@app.get("/users/{user_id}")
async def get_user(user_id: int = Path(..., ge=1)):
    """Get a specific user by ID."""
    user = next((u for u in users_db if u["user_id"] == user_id), None)
    if not user:
        raise HTTPException(status_code=404, detail="User not found")
    return {"data": user}

@app.get("/users/{user_id}/orders")
async def get_user_orders(
    user_id: int = Path(..., ge=1),
    limit: int = Query(default=50, ge=1, le=200)
):
    """Get orders for a specific user."""
    user = next((u for u in users_db if u["user_id"] == user_id), None)
    if not user:
        raise HTTPException(status_code=404, detail="User not found")
    
    user_orders = [o for o in orders_db if o["customer_id"] == user_id][:limit]
    
    return {
        "data": user_orders,
        "user_id": user_id,
        "total_orders": len([o for o in orders_db if o["customer_id"] == user_id])
    }

# Orders endpoints
@app.get("/orders")
async def get_orders(
    limit: int = Query(default=100, ge=1, le=1000),
    offset: int = Query(default=0, ge=0),
    status: Optional[str] = Query(default=None),
    customer_id: Optional[int] = Query(default=None),
    date_from: Optional[str] = Query(default=None),
    date_to: Optional[str] = Query(default=None),
    min_amount: Optional[float] = Query(default=None),
    max_amount: Optional[float] = Query(default=None)
):
    """Get orders with filtering and pagination."""
    filtered_orders = orders_db.copy()
    
    # Apply filters
    if status:
        filtered_orders = [o for o in filtered_orders if o["status"] == status]
    
    if customer_id:
        filtered_orders = [o for o in filtered_orders if o["customer_id"] == customer_id]
    
    if date_from:
        filtered_orders = [o for o in filtered_orders if o["order_date"] >= date_from]
    
    if date_to:
        filtered_orders = [o for o in filtered_orders if o["order_date"] <= date_to]
    
    if min_amount:
        filtered_orders = [o for o in filtered_orders if o["total_amount"] >= min_amount]
    
    if max_amount:
        filtered_orders = [o for o in filtered_orders if o["total_amount"] <= max_amount]
    
    # Apply pagination
    total = len(filtered_orders)
    paginated_orders = filtered_orders[offset:offset + limit]
    
    return {
        "data": paginated_orders,
        "pagination": {
            "total": total,
            "limit": limit,
            "offset": offset,
            "has_next": offset + limit < total,
            "has_previous": offset > 0
        }
    }

@app.get("/orders/{order_id}")
async def get_order(order_id: int = Path(..., ge=1)):
    """Get a specific order by ID."""
    order = next((o for o in orders_db if o["order_id"] == order_id), None)
    if not order:
        raise HTTPException(status_code=404, detail="Order not found")
    return {"data": order}

# Products endpoints
@app.get("/products")
async def get_products(
    limit: int = Query(default=100, ge=1, le=500),
    offset: int = Query(default=0, ge=0),
    category: Optional[str] = Query(default=None),
    active_only: bool = Query(default=False),
    min_price: Optional[float] = Query(default=None),
    max_price: Optional[float] = Query(default=None),
    in_stock: Optional[bool] = Query(default=None)
):
    """Get products with filtering and pagination."""
    filtered_products = products_db.copy()
    
    # Apply filters
    if category:
        filtered_products = [p for p in filtered_products if p["category"] == category]
    
    if active_only:
        filtered_products = [p for p in filtered_products if p["is_active"]]
    
    if min_price:
        filtered_products = [p for p in filtered_products if p["price"] >= min_price]
    
    if max_price:
        filtered_products = [p for p in filtered_products if p["price"] <= max_price]
    
    if in_stock is not None:
        if in_stock:
            filtered_products = [p for p in filtered_products if p["stock_quantity"] > 0]
        else:
            filtered_products = [p for p in filtered_products if p["stock_quantity"] == 0]
    
    # Apply pagination
    total = len(filtered_products)
    paginated_products = filtered_products[offset:offset + limit]
    
    return {
        "data": paginated_products,
        "pagination": {
            "total": total,
            "limit": limit,
            "offset": offset,
            "has_next": offset + limit < total,
            "has_previous": offset > 0
        }
    }

@app.get("/products/{product_id}")
async def get_product(product_id: int = Path(..., ge=1)):
    """Get a specific product by ID."""
    product = next((p for p in products_db if p["product_id"] == product_id), None)
    if not product:
        raise HTTPException(status_code=404, detail="Product not found")
    return {"data": product}

@app.get("/products/categories")
async def get_product_categories():
    """Get all product categories."""
    categories = list(set(p["category"] for p in products_db))
    category_stats = {}
    for category in categories:
        products_in_category = [p for p in products_db if p["category"] == category]
        category_stats[category] = {
            "count": len(products_in_category),
            "active_count": len([p for p in products_in_category if p["is_active"]]),
            "avg_price": round(sum(p["price"] for p in products_in_category) / len(products_in_category), 2)
        }
    
    return {
        "categories": categories,
        "statistics": category_stats
    }

# Analytics endpoints
@app.get("/analytics")
async def get_analytics(
    date_from: Optional[str] = Query(default=None),
    date_to: Optional[str] = Query(default=None),
    metrics: Optional[str] = Query(default=None, description="Comma-separated list of metrics")
):
    """Get analytics data."""
    filtered_analytics = analytics_db.copy()
    
    # Apply date filters
    if date_from:
        filtered_analytics = [a for a in filtered_analytics if a["date"] >= date_from]
    
    if date_to:
        filtered_analytics = [a for a in filtered_analytics if a["date"] <= date_to]
    
    # Filter metrics if specified
    if metrics:
        requested_metrics = [m.strip() for m in metrics.split(",")]
        filtered_data = []
        for item in filtered_analytics:
            filtered_item = {"date": item["date"]}
            for metric in requested_metrics:
                if metric in item:
                    filtered_item[metric] = item[metric]
            filtered_data.append(filtered_item)
        filtered_analytics = filtered_data
    
    return {
        "data": filtered_analytics,
        "summary": {
            "total_records": len(filtered_analytics),
            "date_range": {
                "from": filtered_analytics[0]["date"] if filtered_analytics else None,
                "to": filtered_analytics[-1]["date"] if filtered_analytics else None
            }
        }
    }

@app.get("/analytics/summary")
async def get_analytics_summary():
    """Get aggregated analytics summary."""
    if not analytics_db:
        return {"message": "No analytics data available"}
    
    total_revenue = sum(a["revenue"] for a in analytics_db)
    total_orders = sum(a["orders_count"] for a in analytics_db)
    total_page_views = sum(a["page_views"] for a in analytics_db)
    avg_conversion_rate = sum(a["conversion_rate"] for a in analytics_db) / len(analytics_db)
    
    return {
        "summary": {
            "total_revenue": round(total_revenue, 2),
            "total_orders": total_orders,
            "total_page_views": total_page_views,
            "avg_conversion_rate": round(avg_conversion_rate, 4),
            "avg_daily_revenue": round(total_revenue / len(analytics_db), 2),
            "avg_daily_orders": round(total_orders / len(analytics_db), 1)
        },
        "period": {
            "days": len(analytics_db),
            "from": analytics_db[0]["date"],
            "to": analytics_db[-1]["date"]
        }
    }

# Utility endpoints for testing
@app.post("/data/regenerate")
async def regenerate_data():
    """Regenerate all test data (for testing purposes)."""
    generate_test_data()
    return {
        "message": "Test data regenerated successfully",
        "counts": {
            "users": len(users_db),
            "orders": len(orders_db),
            "products": len(products_db),
            "analytics": len(analytics_db)
        }
    }

@app.get("/data/stats")
async def get_data_stats():
    """Get statistics about the test data."""
    return {
        "counts": {
            "users": len(users_db),
            "orders": len(orders_db),
            "products": len(products_db),
            "analytics": len(analytics_db)
        },
        "users": {
            "active": len([u for u in users_db if u["is_active"]]),
            "inactive": len([u for u in users_db if not u["is_active"]]),
            "countries": len(set(u["profile"]["country"] for u in users_db))
        },
        "orders": {
            "by_status": {
                status: len([o for o in orders_db if o["status"] == status])
                for status in set(o["status"] for o in orders_db)
            },
            "total_revenue": round(sum(o["total_amount"] for o in orders_db), 2)
        },
        "products": {
            "active": len([p for p in products_db if p["is_active"]]),
            "categories": len(set(p["category"] for p in products_db)),
            "in_stock": len([p for p in products_db if p["stock_quantity"] > 0])
        }
    }

# Simulate slow endpoints for testing
@app.get("/slow/users")
async def get_users_slow(delay: int = Query(default=2, ge=1, le=10)):
    """Slow users endpoint for testing timeouts."""
    await asyncio.sleep(delay)
    return {"data": users_db[:10], "delay": delay}

# Error simulation endpoints
@app.get("/error/{error_code}")
async def simulate_error(error_code: int = Path(..., ge=400, le=599)):
    """Simulate HTTP errors for testing."""
    error_messages = {
        400: "Bad Request - Invalid parameters",
        401: "Unauthorized - Authentication required",
        403: "Forbidden - Access denied",
        404: "Not Found - Resource does not exist",
        500: "Internal Server Error - Something went wrong",
        502: "Bad Gateway - Upstream server error",
        503: "Service Unavailable - Server is temporarily unavailable"
    }
    
    message = error_messages.get(error_code, f"HTTP Error {error_code}")
    raise HTTPException(status_code=error_code, detail=message)

if __name__ == "__main__":
    print(f"Starting Mock API Server on {API_HOST}:{API_PORT}")
    print(f"Documentation available at http://{API_HOST}:{API_PORT}/docs")
    
    uvicorn.run(
        "app:app",
        host=API_HOST,
        port=API_PORT,
        log_level=LOG_LEVEL,
        reload=False
    )