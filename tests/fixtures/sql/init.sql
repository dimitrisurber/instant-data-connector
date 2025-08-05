-- PostgreSQL initialization script for FDW testing
-- This script sets up the test database with sample data and FDW extensions

-- Enable required extensions
CREATE EXTENSION IF NOT EXISTS postgres_fdw;
CREATE EXTENSION IF NOT EXISTS pg_stat_statements;
CREATE EXTENSION IF NOT EXISTS btree_gin;
CREATE EXTENSION IF NOT EXISTS btree_gist;

-- Create test schemas
CREATE SCHEMA IF NOT EXISTS fdw_test;
CREATE SCHEMA IF NOT EXISTS staging;

-- Create users table
CREATE TABLE IF NOT EXISTS users (
    user_id SERIAL PRIMARY KEY,
    username VARCHAR(50) UNIQUE NOT NULL,
    first_name VARCHAR(50),
    last_name VARCHAR(50),
    email VARCHAR(255) UNIQUE NOT NULL,
    registration_date DATE DEFAULT CURRENT_DATE,
    last_login TIMESTAMP,
    is_active BOOLEAN DEFAULT TRUE,
    metadata JSONB,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Create orders table
CREATE TABLE IF NOT EXISTS orders (
    order_id SERIAL PRIMARY KEY,
    customer_id INTEGER REFERENCES users(user_id),
    order_number VARCHAR(50) UNIQUE NOT NULL,
    status VARCHAR(20) NOT NULL DEFAULT 'pending',
    total_amount DECIMAL(12,2) NOT NULL,
    order_date TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
    items INTEGER DEFAULT 1,
    shipping_address JSONB,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Create products table
CREATE TABLE IF NOT EXISTS products (
    product_id SERIAL PRIMARY KEY,
    sku VARCHAR(50) UNIQUE NOT NULL,
    name VARCHAR(255) NOT NULL,
    description TEXT,
    category VARCHAR(100),
    price DECIMAL(10,2) NOT NULL,
    cost DECIMAL(10,2),
    weight DECIMAL(8,3),
    stock_quantity INTEGER DEFAULT 0,
    is_active BOOLEAN DEFAULT TRUE,
    attributes JSONB,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Create order_items table for normalized order structure
CREATE TABLE IF NOT EXISTS order_items (
    order_item_id SERIAL PRIMARY KEY,
    order_id INTEGER REFERENCES orders(order_id) ON DELETE CASCADE,
    product_id INTEGER REFERENCES products(product_id),
    quantity INTEGER NOT NULL DEFAULT 1,
    unit_price DECIMAL(10,2) NOT NULL,
    total_price DECIMAL(12,2) GENERATED ALWAYS AS (quantity * unit_price) STORED,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Create indexes for better performance
CREATE INDEX IF NOT EXISTS idx_users_email ON users(email);
CREATE INDEX IF NOT EXISTS idx_users_active ON users(is_active);
CREATE INDEX IF NOT EXISTS idx_users_registration_date ON users(registration_date);
CREATE INDEX IF NOT EXISTS idx_orders_customer_id ON orders(customer_id);
CREATE INDEX IF NOT EXISTS idx_orders_status ON orders(status);
CREATE INDEX IF NOT EXISTS idx_orders_date ON orders(order_date);
CREATE INDEX IF NOT EXISTS idx_products_category ON products(category);
CREATE INDEX IF NOT EXISTS idx_products_active ON products(is_active);
CREATE INDEX IF NOT EXISTS idx_order_items_order_id ON order_items(order_id);
CREATE INDEX IF NOT EXISTS idx_order_items_product_id ON order_items(product_id);

-- Insert sample users data
INSERT INTO users (username, first_name, last_name, email, registration_date, last_login, is_active, metadata) VALUES
('alice_smith', 'Alice', 'Smith', 'alice.smith@example.com', '2024-01-15', '2024-08-01 10:30:00', true, '{"preferences": {"theme": "dark", "notifications": true}}'),
('bob_jones', 'Bob', 'Jones', 'bob.jones@example.com', '2024-01-20', '2024-08-02 14:15:00', true, '{"preferences": {"theme": "light", "notifications": false}}'),
('charlie_brown', 'Charlie', 'Brown', 'charlie.brown@example.com', '2024-02-01', '2024-07-30 09:45:00', true, '{"preferences": {"theme": "auto", "notifications": true}}'),
('diana_wilson', 'Diana', 'Wilson', 'diana.wilson@example.com', '2024-02-10', '2024-08-01 16:20:00', false, '{"preferences": {"theme": "light", "notifications": true}}'),
('eve_davis', 'Eve', 'Davis', 'eve.davis@example.com', '2024-02-15', '2024-08-03 11:10:00', true, '{"preferences": {"theme": "dark", "notifications": false}}'),
('frank_miller', 'Frank', 'Miller', 'frank.miller@example.com', '2024-03-01', '2024-07-28 13:45:00', true, '{"preferences": {"theme": "light", "notifications": true}}'),
('grace_taylor', 'Grace', 'Taylor', 'grace.taylor@example.com', '2024-03-10', '2024-08-02 08:30:00', true, '{"preferences": {"theme": "dark", "notifications": true}}'),
('henry_anderson', 'Henry', 'Anderson', 'henry.anderson@example.com', '2024-03-15', NULL, false, '{"preferences": {"theme": "auto", "notifications": false}}'),
('ivy_thomas', 'Ivy', 'Thomas', 'ivy.thomas@example.com', '2024-03-20', '2024-08-01 12:15:00', true, '{"preferences": {"theme": "light", "notifications": true}}'),
('jack_jackson', 'Jack', 'Jackson', 'jack.jackson@example.com', '2024-04-01', '2024-08-03 15:30:00', true, '{"preferences": {"theme": "dark", "notifications": false}}')
ON CONFLICT (username) DO NOTHING;

-- Generate additional test users
INSERT INTO users (username, first_name, last_name, email, registration_date, last_login, is_active)
SELECT 
    'user_' || generate_series,
    'FirstName' || generate_series,
    'LastName' || generate_series,
    'user' || generate_series || '@example.com',
    CURRENT_DATE - (random() * 365)::int,
    CASE 
        WHEN random() > 0.2 THEN CURRENT_TIMESTAMP - (random() * 30)::int * INTERVAL '1 day'
        ELSE NULL 
    END,
    random() > 0.1
FROM generate_series(11, 100)
ON CONFLICT (username) DO NOTHING;

-- Insert sample products
INSERT INTO products (sku, name, description, category, price, cost, weight, stock_quantity, is_active, attributes) VALUES
('LAPTOP001', 'Gaming Laptop Pro', 'High-performance gaming laptop with RTX graphics', 'Electronics', 1299.99, 950.00, 2.5, 25, true, '{"brand": "TechCorp", "warranty": "2 years", "color": "black"}'),
('MOUSE001', 'Wireless Gaming Mouse', 'Precision wireless mouse for gaming', 'Electronics', 79.99, 45.00, 0.15, 150, true, '{"brand": "TechCorp", "warranty": "1 year", "color": "black"}'),
('KEYBOARD001', 'Mechanical Keyboard', 'RGB mechanical keyboard with Cherry MX switches', 'Electronics', 149.99, 85.00, 1.2, 75, true, '{"brand": "TechCorp", "warranty": "2 years", "switches": "Cherry MX Blue"}'),
('MONITOR001', '4K Gaming Monitor', '32-inch 4K monitor with 144Hz refresh rate', 'Electronics', 599.99, 400.00, 8.5, 40, true, '{"brand": "DisplayTech", "warranty": "3 years", "size": "32 inch"}'),
('CHAIR001', 'Ergonomic Office Chair', 'Comfortable ergonomic chair for long work sessions', 'Furniture', 299.99, 180.00, 15.0, 20, true, '{"brand": "ComfortSeats", "warranty": "5 years", "color": "black"}'),
('DESK001', 'Standing Desk', 'Height-adjustable standing desk', 'Furniture', 449.99, 280.00, 35.0, 15, true, '{"brand": "WorkSpace", "warranty": "3 years", "material": "wood"}'),
('HEADSET001', 'Wireless Headset', 'Noise-cancelling wireless headset', 'Electronics', 199.99, 120.00, 0.35, 80, true, '{"brand": "AudioTech", "warranty": "2 years", "color": "black"}'),
('WEBCAM001', '4K Webcam', 'Ultra HD webcam for streaming', 'Electronics', 129.99, 75.00, 0.25, 60, true, '{"brand": "StreamTech", "warranty": "1 year", "resolution": "4K"}'),
('TABLET001', 'Professional Tablet', '12-inch tablet for creative work', 'Electronics', 699.99, 450.00, 0.8, 30, true, '{"brand": "CreativeTech", "warranty": "2 years", "storage": "256GB"}'),
('SPEAKER001', 'Bluetooth Speaker', 'Portable wireless speaker', 'Electronics', 89.99, 50.00, 0.6, 100, true, '{"brand": "AudioTech", "warranty": "1 year", "battery": "12 hours"}')
ON CONFLICT (sku) DO NOTHING;

-- Insert sample orders
INSERT INTO orders (customer_id, order_number, status, total_amount, order_date, items, shipping_address) VALUES
(1, 'ORD-2024-001', 'completed', 1379.98, '2024-07-01 10:30:00', 2, '{"street": "123 Main St", "city": "New York", "state": "NY", "zip": "10001"}'),
(2, 'ORD-2024-002', 'completed', 149.99, '2024-07-02 14:15:00', 1, '{"street": "456 Oak Ave", "city": "Los Angeles", "state": "CA", "zip": "90210"}'),
(3, 'ORD-2024-003', 'pending', 749.98, '2024-07-05 09:45:00', 2, '{"street": "789 Pine Rd", "city": "Chicago", "state": "IL", "zip": "60601"}'),
(1, 'ORD-2024-004', 'shipped', 299.99, '2024-07-08 16:20:00', 1, '{"street": "123 Main St", "city": "New York", "state": "NY", "zip": "10001"}'),
(4, 'ORD-2024-005', 'completed', 829.97, '2024-07-10 11:10:00', 3, '{"street": "321 Elm St", "city": "Miami", "state": "FL", "zip": "33101"}'),
(5, 'ORD-2024-006', 'cancelled', 199.99, '2024-07-12 13:45:00', 1, '{"street": "654 Maple Dr", "city": "Seattle", "state": "WA", "zip": "98101"}'),
(6, 'ORD-2024-007', 'completed', 1149.97, '2024-07-15 08:30:00', 3, '{"street": "987 Cedar Ln", "city": "Denver", "state": "CO", "zip": "80202"}'),
(2, 'ORD-2024-008', 'processing', 449.99, '2024-07-18 12:15:00', 1, '{"street": "456 Oak Ave", "city": "Los Angeles", "state": "CA", "zip": "90210"}'),
(7, 'ORD-2024-009', 'completed', 89.99, '2024-07-20 15:30:00', 1, '{"street": "147 Birch St", "city": "Boston", "state": "MA", "zip": "02101"}'),
(8, 'ORD-2024-010', 'shipped', 929.97, '2024-07-22 09:00:00', 2, '{"street": "258 Spruce Ave", "city": "Austin", "state": "TX", "zip": "73301"}')
ON CONFLICT (order_number) DO NOTHING;

-- Insert order items
INSERT INTO order_items (order_id, product_id, quantity, unit_price) VALUES
-- Order 1: Gaming Laptop + Mouse
(1, 1, 1, 1299.99),
(1, 2, 1, 79.99),
-- Order 2: Mechanical Keyboard
(2, 3, 1, 149.99),
-- Order 3: 4K Monitor + Gaming Mouse
(3, 4, 1, 599.99),
(3, 2, 1, 149.99),
-- Order 4: Ergonomic Chair
(4, 5, 1, 299.99),
-- Order 5: Headset + Webcam + Bluetooth Speaker
(5, 7, 1, 199.99),
(5, 8, 1, 129.99),
(5, 10, 1, 89.99),
-- Order 6: Headset (cancelled)
(6, 7, 1, 199.99),
-- Order 7: Gaming Laptop + Keyboard + Mouse
(7, 1, 1, 1299.99),
(7, 3, 1, 149.99),
(7, 2, 1, 79.99),
-- Order 8: Standing Desk
(8, 6, 1, 449.99),
-- Order 9: Bluetooth Speaker
(9, 10, 1, 89.99),
-- Order 10: Professional Tablet + Gaming Mouse
(10, 9, 1, 699.99),
(10, 2, 1, 79.99)
ON CONFLICT DO NOTHING;

-- Generate additional random orders for testing
INSERT INTO orders (customer_id, order_number, status, total_amount, order_date, items)
SELECT 
    (random() * 99 + 1)::int,  -- Random customer_id between 1-100
    'ORD-2024-' || LPAD((10 + generate_series)::text, 3, '0'),
    CASE 
        WHEN random() < 0.6 THEN 'completed'
        WHEN random() < 0.8 THEN 'shipped'
        WHEN random() < 0.9 THEN 'processing'
        WHEN random() < 0.95 THEN 'pending'
        ELSE 'cancelled'
    END,
    (random() * 2000 + 50)::numeric(12,2), -- Random amount between $50-$2050
    CURRENT_DATE - (random() * 180)::int + (random() * 24)::int * INTERVAL '1 hour',
    (random() * 5 + 1)::int  -- 1-5 items
FROM generate_series(1, 190)  -- Generate 190 more orders for total of 200
ON CONFLICT (order_number) DO NOTHING;

-- Create materialized view for order statistics
CREATE MATERIALIZED VIEW IF NOT EXISTS order_stats AS
SELECT 
    DATE_TRUNC('month', order_date) as month,
    status,
    COUNT(*) as order_count,
    SUM(total_amount) as total_revenue,
    AVG(total_amount) as avg_order_value,
    MIN(total_amount) as min_order_value,
    MAX(total_amount) as max_order_value
FROM orders
GROUP BY DATE_TRUNC('month', order_date), status
ORDER BY month DESC, status;

-- Create a view for user activity
CREATE OR REPLACE VIEW user_activity AS
SELECT 
    u.user_id,
    u.username,
    u.first_name,
    u.last_name,
    u.email,
    u.registration_date,
    u.last_login,
    u.is_active,
    COUNT(o.order_id) as total_orders,
    SUM(o.total_amount) as total_spent,
    AVG(o.total_amount) as avg_order_value,
    MAX(o.order_date) as last_order_date
FROM users u
LEFT JOIN orders o ON u.user_id = o.customer_id
GROUP BY u.user_id, u.username, u.first_name, u.last_name, u.email, 
         u.registration_date, u.last_login, u.is_active;

-- Create function for order processing
CREATE OR REPLACE FUNCTION process_order(order_id_param INT)
RETURNS JSON AS $$
DECLARE
    result JSON;
BEGIN
    UPDATE orders 
    SET status = 'processing', updated_at = CURRENT_TIMESTAMP
    WHERE order_id = order_id_param AND status = 'pending';
    
    SELECT json_build_object(
        'order_id', order_id_param,
        'status', 'processing',
        'processed_at', CURRENT_TIMESTAMP
    ) INTO result;
    
    RETURN result;
END;
$$ LANGUAGE plpgsql;

-- Create trigger to update updated_at timestamp
CREATE OR REPLACE FUNCTION update_updated_at_column()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = CURRENT_TIMESTAMP;
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

-- Apply triggers to tables
DROP TRIGGER IF EXISTS update_users_updated_at ON users;
CREATE TRIGGER update_users_updated_at 
    BEFORE UPDATE ON users 
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

DROP TRIGGER IF EXISTS update_orders_updated_at ON orders;
CREATE TRIGGER update_orders_updated_at 
    BEFORE UPDATE ON orders 
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

DROP TRIGGER IF EXISTS update_products_updated_at ON products;
CREATE TRIGGER update_products_updated_at 
    BEFORE UPDATE ON products 
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

-- Grant permissions for test user
GRANT ALL PRIVILEGES ON ALL TABLES IN SCHEMA public TO test_user;
GRANT ALL PRIVILEGES ON ALL SEQUENCES IN SCHEMA public TO test_user;
GRANT ALL PRIVILEGES ON ALL FUNCTIONS IN SCHEMA public TO test_user;
GRANT USAGE ON SCHEMA fdw_test TO test_user;
GRANT USAGE ON SCHEMA staging TO test_user;

-- Refresh materialized view
REFRESH MATERIALIZED VIEW order_stats;

-- Show completion message
DO $$
BEGIN
    RAISE NOTICE 'Test database initialization completed successfully!';
    RAISE NOTICE 'Created % users, % products, % orders', 
        (SELECT COUNT(*) FROM users),
        (SELECT COUNT(*) FROM products),
        (SELECT COUNT(*) FROM orders);
END $$;