-- MySQL initialization script for FDW testing
-- This script sets up the MySQL test database with sample data

USE test_mysql_db;

-- Create legacy customers table
CREATE TABLE IF NOT EXISTS legacy_customers (
    customer_id INT AUTO_INCREMENT PRIMARY KEY,
    first_name VARCHAR(50) NOT NULL,
    last_name VARCHAR(50) NOT NULL,
    email VARCHAR(100) UNIQUE NOT NULL,
    phone VARCHAR(20),
    address_line1 VARCHAR(100),
    address_line2 VARCHAR(100),
    city VARCHAR(50),
    state VARCHAR(30),
    zip_code VARCHAR(10),
    country VARCHAR(50) DEFAULT 'USA',
    date_created DATE DEFAULT (CURRENT_DATE),
    last_purchase_date DATE,
    total_purchases DECIMAL(12,2) DEFAULT 0.00,
    is_active BOOLEAN DEFAULT TRUE,
    customer_tier ENUM('bronze', 'silver', 'gold', 'platinum') DEFAULT 'bronze',
    notes TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP
);

-- Create inventory table
CREATE TABLE IF NOT EXISTS inventory (
    inventory_id INT AUTO_INCREMENT PRIMARY KEY,
    product_id INT NOT NULL,
    warehouse_id INT NOT NULL,
    quantity_on_hand INT NOT NULL DEFAULT 0,
    quantity_reserved INT DEFAULT 0,
    quantity_available INT GENERATED ALWAYS AS (quantity_on_hand - quantity_reserved) STORED,
    reorder_point INT DEFAULT 0,
    max_stock_level INT DEFAULT 1000,
    unit_cost DECIMAL(10,2),
    last_count_date DATE,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
    UNIQUE KEY unique_product_warehouse (product_id, warehouse_id)
);

-- Create warehouses table
CREATE TABLE IF NOT EXISTS warehouses (
    warehouse_id INT AUTO_INCREMENT PRIMARY KEY,
    warehouse_code VARCHAR(10) UNIQUE NOT NULL,
    name VARCHAR(100) NOT NULL,
    address_line1 VARCHAR(100),
    address_line2 VARCHAR(100),
    city VARCHAR(50),
    state VARCHAR(30),
    zip_code VARCHAR(10),
    country VARCHAR(50) DEFAULT 'USA',
    manager_name VARCHAR(100),
    phone VARCHAR(20),
    email VARCHAR(100),
    is_active BOOLEAN DEFAULT TRUE,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP
);

-- Create suppliers table
CREATE TABLE IF NOT EXISTS suppliers (
    supplier_id INT AUTO_INCREMENT PRIMARY KEY,
    supplier_code VARCHAR(20) UNIQUE NOT NULL,
    company_name VARCHAR(100) NOT NULL,
    contact_name VARCHAR(100),
    contact_email VARCHAR(100),
    contact_phone VARCHAR(20),
    address_line1 VARCHAR(100),
    address_line2 VARCHAR(100),
    city VARCHAR(50),
    state VARCHAR(30),
    zip_code VARCHAR(10),
    country VARCHAR(50) DEFAULT 'USA',
    payment_terms VARCHAR(50),
    is_active BOOLEAN DEFAULT TRUE,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP
);

-- Create product_suppliers junction table
CREATE TABLE IF NOT EXISTS product_suppliers (
    product_supplier_id INT AUTO_INCREMENT PRIMARY KEY,
    product_id INT NOT NULL,
    supplier_id INT NOT NULL,
    supplier_sku VARCHAR(50),
    lead_time_days INT DEFAULT 7,
    minimum_order_quantity INT DEFAULT 1,
    cost_per_unit DECIMAL(10,2),
    is_preferred BOOLEAN DEFAULT FALSE,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
    UNIQUE KEY unique_product_supplier (product_id, supplier_id),
    FOREIGN KEY (supplier_id) REFERENCES suppliers(supplier_id)
);

-- Create indexes for better performance
CREATE INDEX idx_legacy_customers_email ON legacy_customers(email);
CREATE INDEX idx_legacy_customers_active ON legacy_customers(is_active);
CREATE INDEX idx_legacy_customers_tier ON legacy_customers(customer_tier);
CREATE INDEX idx_legacy_customers_created ON legacy_customers(date_created);
CREATE INDEX idx_inventory_product ON inventory(product_id);
CREATE INDEX idx_inventory_warehouse ON inventory(warehouse_id);
CREATE INDEX idx_inventory_quantity ON inventory(quantity_on_hand);
CREATE INDEX idx_warehouses_code ON warehouses(warehouse_code);
CREATE INDEX idx_warehouses_active ON warehouses(is_active);
CREATE INDEX idx_suppliers_code ON suppliers(supplier_code);
CREATE INDEX idx_suppliers_active ON suppliers(is_active);

-- Insert sample warehouses
INSERT INTO warehouses (warehouse_code, name, address_line1, city, state, zip_code, manager_name, phone, email, is_active) VALUES
('WH001', 'East Coast Distribution Center', '1000 Industrial Blvd', 'Atlanta', 'GA', '30309', 'John Smith', '404-555-0101', 'john.smith@warehouse.com', TRUE),
('WH002', 'West Coast Distribution Center', '2000 Logistics Way', 'Los Angeles', 'CA', '90210', 'Jane Doe', '310-555-0102', 'jane.doe@warehouse.com', TRUE),
('WH003', 'Midwest Distribution Center', '3000 Commerce Dr', 'Chicago', 'IL', '60601', 'Mike Johnson', '312-555-0103', 'mike.johnson@warehouse.com', TRUE),
('WH004', 'Texas Distribution Center', '4000 Supply Chain Rd', 'Dallas', 'TX', '75201', 'Sarah Wilson', '214-555-0104', 'sarah.wilson@warehouse.com', TRUE),
('WH005', 'Northwest Distribution Center', '5000 Freight Ave', 'Seattle', 'WA', '98101', 'David Brown', '206-555-0105', 'david.brown@warehouse.com', FALSE)
ON DUPLICATE KEY UPDATE name=VALUES(name);

-- Insert sample suppliers
INSERT INTO suppliers (supplier_code, company_name, contact_name, contact_email, contact_phone, address_line1, city, state, zip_code, payment_terms, is_active) VALUES
('SUP001', 'TechCorp Manufacturing', 'Robert Chen', 'robert.chen@techcorp.com', '555-0201', '100 Technology Dr', 'San Jose', 'CA', '95110', 'Net 30', TRUE),
('SUP002', 'Global Electronics Ltd', 'Maria Garcia', 'maria.garcia@globalelec.com', '555-0202', '200 Circuit Blvd', 'Austin', 'TX', '78701', 'Net 45', TRUE),
('SUP003', 'Furniture Masters Inc', 'James Anderson', 'james.anderson@furnituremasters.com', '555-0203', '300 Woodwork Way', 'Grand Rapids', 'MI', '49503', 'Net 30', TRUE),
('SUP004', 'Audio Visual Solutions', 'Lisa Taylor', 'lisa.taylor@avsolutions.com', '555-0204', '400 Media Rd', 'Nashville', 'TN', '37201', 'Net 15', TRUE),
('SUP005', 'Creative Tech Supplies', 'Kevin White', 'kevin.white@creativeteach.com', '555-0205', '500 Innovation St', 'Portland', 'OR', '97201', 'Net 30', FALSE)
ON DUPLICATE KEY UPDATE company_name=VALUES(company_name);

-- Insert sample legacy customers
INSERT INTO legacy_customers (first_name, last_name, email, phone, address_line1, city, state, zip_code, date_created, last_purchase_date, total_purchases, is_active, customer_tier, notes) VALUES
('Alice', 'Johnson', 'alice.johnson@email.com', '555-1001', '123 Maple Street', 'New York', 'NY', '10001', '2023-01-15', '2024-07-20', 2850.75, TRUE, 'gold', 'VIP customer, prefers express shipping'),
('Bob', 'Williams', 'bob.williams@email.com', '555-1002', '456 Oak Avenue', 'Los Angeles', 'CA', '90210', '2023-02-10', '2024-07-18', 1245.50, TRUE, 'silver', 'Regular customer, bulk order discounts'),
('Carol', 'Davis', 'carol.davis@email.com', '555-1003', '789 Pine Road', 'Chicago', 'IL', '60601', '2023-03-05', '2024-06-15', 750.25, TRUE, 'bronze', 'Price-sensitive, looks for deals'),
('David', 'Miller', 'david.miller@email.com', '555-1004', '321 Elm Street', 'Houston', 'TX', '77001', '2023-03-20', '2024-07-22', 3420.00, TRUE, 'platinum', 'Enterprise customer, dedicated account manager'),
('Eve', 'Wilson', 'eve.wilson@email.com', '555-1005', '654 Cedar Lane', 'Phoenix', 'AZ', '85001', '2023-04-12', '2024-05-10', 425.75, FALSE, 'bronze', 'Inactive customer, last purchase over 60 days ago'),
('Frank', 'Moore', 'frank.moore@email.com', '555-1006', '987 Birch Drive', 'Philadelphia', 'PA', '19101', '2023-05-08', '2024-07-25', 1875.25, TRUE, 'silver', 'Frequent buyer, newsletter subscriber'),
('Grace', 'Taylor', 'grace.taylor@email.com', '555-1007', '147 Spruce Way', 'San Antonio', 'TX', '78201', '2023-06-15', '2024-07-15', 965.50, TRUE, 'bronze', 'First-time bulk order customer'),
('Henry', 'Anderson', 'henry.anderson@email.com', '555-1008', '258 Willow Court', 'San Diego', 'CA', '92101', '2023-07-01', NULL, 0.00, TRUE, 'bronze', 'New customer, no purchases yet'),
('Ivy', 'Thomas', 'ivy.thomas@email.com', '555-1009', '369 Poplar Street', 'Dallas', 'TX', '75201', '2023-07-20', '2024-07-28', 2100.00, TRUE, 'gold', 'Referral customer, high lifetime value'),
('Jack', 'Jackson', 'jack.jackson@email.com', '555-1010', '741 Magnolia Blvd', 'San Jose', 'CA', '95110', '2023-08-10', '2024-07-30', 625.75, TRUE, 'bronze', 'Tech enthusiast, early adopter')
ON DUPLICATE KEY UPDATE email=VALUES(email);

-- Generate additional test customers
INSERT INTO legacy_customers (first_name, last_name, email, phone, address_line1, city, state, zip_code, date_created, last_purchase_date, total_purchases, is_active, customer_tier)
SELECT 
    CONCAT('Customer', n),
    CONCAT('LastName', n),
    CONCAT('customer', n, '@testdomain.com'),
    CONCAT('555-', LPAD(n, 4, '0')),
    CONCAT(n * 100, ' Test Street'),
    CASE 
        WHEN n % 10 = 1 THEN 'New York'
        WHEN n % 10 = 2 THEN 'Los Angeles'
        WHEN n % 10 = 3 THEN 'Chicago'
        WHEN n % 10 = 4 THEN 'Houston'
        WHEN n % 10 = 5 THEN 'Phoenix'
        WHEN n % 10 = 6 THEN 'Philadelphia'
        WHEN n % 10 = 7 THEN 'San Antonio'
        WHEN n % 10 = 8 THEN 'San Diego'
        WHEN n % 10 = 9 THEN 'Dallas'
        ELSE 'Miami'
    END,
    CASE 
        WHEN n % 10 = 1 THEN 'NY'
        WHEN n % 10 = 2 THEN 'CA'
        WHEN n % 10 = 3 THEN 'IL'
        WHEN n % 10 = 4 THEN 'TX'
        WHEN n % 10 = 5 THEN 'AZ'
        WHEN n % 10 = 6 THEN 'PA'
        WHEN n % 10 = 7 THEN 'TX'
        WHEN n % 10 = 8 THEN 'CA'
        WHEN n % 10 = 9 THEN 'TX'
        ELSE 'FL'
    END,
    LPAD(n * 123 % 99999, 5, '0'),
    DATE_SUB(CURDATE(), INTERVAL FLOOR(RAND() * 365) DAY),
    CASE WHEN RAND() > 0.2 THEN DATE_SUB(CURDATE(), INTERVAL FLOOR(RAND() * 90) DAY) ELSE NULL END,
    ROUND(RAND() * 5000, 2),
    RAND() > 0.1,
    CASE 
        WHEN RAND() < 0.1 THEN 'platinum'
        WHEN RAND() < 0.3 THEN 'gold'
        WHEN RAND() < 0.6 THEN 'silver'
        ELSE 'bronze'
    END
FROM (
    SELECT a.N + b.N * 10 + 11 as n
    FROM (SELECT 0 AS N UNION ALL SELECT 1 UNION ALL SELECT 2 UNION ALL SELECT 3 UNION ALL SELECT 4 UNION ALL SELECT 5 UNION ALL SELECT 6 UNION ALL SELECT 7 UNION ALL SELECT 8 UNION ALL SELECT 9) a
    CROSS JOIN (SELECT 0 AS N UNION ALL SELECT 1 UNION ALL SELECT 2 UNION ALL SELECT 3 UNION ALL SELECT 4 UNION ALL SELECT 5 UNION ALL SELECT 6 UNION ALL SELECT 7 UNION ALL SELECT 8 UNION ALL SELECT 9) b
    WHERE a.N + b.N * 10 + 11 <= 100
) numbers
ON DUPLICATE KEY UPDATE email=VALUES(email);

-- Insert product supplier relationships
INSERT INTO product_suppliers (product_id, supplier_id, supplier_sku, lead_time_days, minimum_order_quantity, cost_per_unit, is_preferred) VALUES
(1, 1, 'TC-LAPTOP-001', 14, 1, 950.00, TRUE),   -- Gaming Laptop from TechCorp
(2, 1, 'TC-MOUSE-001', 7, 10, 45.00, TRUE),    -- Gaming Mouse from TechCorp  
(3, 1, 'TC-KEYBOARD-001', 10, 5, 85.00, TRUE), -- Mechanical Keyboard from TechCorp
(4, 2, 'GE-MONITOR-001', 21, 1, 400.00, TRUE), -- 4K Monitor from Global Electronics
(5, 3, 'FM-CHAIR-001', 28, 1, 180.00, TRUE),   -- Office Chair from Furniture Masters
(6, 3, 'FM-DESK-001', 35, 1, 280.00, TRUE),    -- Standing Desk from Furniture Masters
(7, 4, 'AV-HEADSET-001', 14, 5, 120.00, TRUE), -- Wireless Headset from Audio Visual
(8, 4, 'AV-WEBCAM-001', 10, 10, 75.00, TRUE),  -- 4K Webcam from Audio Visual
(9, 5, 'CT-TABLET-001', 21, 1, 450.00, FALSE), -- Professional Tablet from Creative Tech (not preferred)
(10, 4, 'AV-SPEAKER-001', 7, 20, 50.00, TRUE)  -- Bluetooth Speaker from Audio Visual
ON DUPLICATE KEY UPDATE supplier_sku=VALUES(supplier_sku);

-- Insert inventory data for all products across all warehouses
INSERT INTO inventory (product_id, warehouse_id, quantity_on_hand, quantity_reserved, reorder_point, max_stock_level, unit_cost, last_count_date)
SELECT 
    p.product_id,
    w.warehouse_id,
    FLOOR(RAND() * 200) + 10 as quantity_on_hand,  -- Random quantity between 10-209
    FLOOR(RAND() * 20) as quantity_reserved,        -- Random reserved between 0-19
    CASE 
        WHEN p.product_id <= 3 THEN 50   -- Electronics have higher reorder points
        WHEN p.product_id <= 6 THEN 20   -- Furniture has medium reorder points
        ELSE 30                          -- Other items
    END as reorder_point,
    CASE 
        WHEN p.product_id <= 3 THEN 500  -- Electronics have higher max stock
        WHEN p.product_id <= 6 THEN 100  -- Furniture has lower max stock
        ELSE 300                         -- Other items
    END as max_stock_level,
    ps.cost_per_unit,
    DATE_SUB(CURDATE(), INTERVAL FLOOR(RAND() * 30) DAY) as last_count_date
FROM (SELECT 1 as product_id UNION SELECT 2 UNION SELECT 3 UNION SELECT 4 UNION SELECT 5 
      UNION SELECT 6 UNION SELECT 7 UNION SELECT 8 UNION SELECT 9 UNION SELECT 10) p
CROSS JOIN warehouses w
LEFT JOIN product_suppliers ps ON p.product_id = ps.product_id AND ps.is_preferred = TRUE
WHERE w.is_active = TRUE
ON DUPLICATE KEY UPDATE quantity_on_hand=VALUES(quantity_on_hand);

-- Create a view for inventory summary
CREATE OR REPLACE VIEW inventory_summary AS
SELECT 
    i.product_id,
    w.warehouse_code,
    w.name as warehouse_name,
    i.quantity_on_hand,
    i.quantity_reserved,
    i.quantity_available,
    i.reorder_point,
    CASE 
        WHEN i.quantity_available <= i.reorder_point THEN 'LOW_STOCK'
        WHEN i.quantity_available <= i.reorder_point * 2 THEN 'MEDIUM_STOCK'
        ELSE 'HIGH_STOCK'
    END as stock_status,
    i.unit_cost,
    i.last_count_date,
    DATEDIFF(CURDATE(), i.last_count_date) as days_since_count
FROM inventory i
JOIN warehouses w ON i.warehouse_id = w.warehouse_id
WHERE w.is_active = TRUE;

-- Create a view for customer analytics
CREATE OR REPLACE VIEW customer_analytics AS
SELECT 
    customer_tier,
    is_active,
    COUNT(*) as customer_count,
    AVG(total_purchases) as avg_total_purchases,
    SUM(total_purchases) as total_tier_revenue,
    MIN(total_purchases) as min_purchases,
    MAX(total_purchases) as max_purchases,
    COUNT(CASE WHEN last_purchase_date >= DATE_SUB(CURDATE(), INTERVAL 30 DAY) THEN 1 END) as recent_buyers,
    COUNT(CASE WHEN last_purchase_date IS NULL THEN 1 END) as never_purchased
FROM legacy_customers
GROUP BY customer_tier, is_active;

-- Create stored procedure for updating customer tier
DELIMITER //
CREATE OR REPLACE PROCEDURE UpdateCustomerTier(IN customer_id_param INT)
BEGIN
    DECLARE purchase_total DECIMAL(12,2);
    DECLARE new_tier ENUM('bronze', 'silver', 'gold', 'platinum');
    
    SELECT total_purchases INTO purchase_total 
    FROM legacy_customers 
    WHERE customer_id = customer_id_param;
    
    IF purchase_total >= 3000 THEN
        SET new_tier = 'platinum';
    ELSEIF purchase_total >= 1500 THEN
        SET new_tier = 'gold';
    ELSEIF purchase_total >= 500 THEN
        SET new_tier = 'silver';
    ELSE
        SET new_tier = 'bronze';
    END IF;
    
    UPDATE legacy_customers 
    SET customer_tier = new_tier, updated_at = CURRENT_TIMESTAMP
    WHERE customer_id = customer_id_param;
    
END //
DELIMITER ;

-- Create function to calculate inventory value
DELIMITER //
CREATE OR REPLACE FUNCTION GetInventoryValue(warehouse_id_param INT)
RETURNS DECIMAL(15,2)
READS SQL DATA
DETERMINISTIC
BEGIN
    DECLARE total_value DECIMAL(15,2) DEFAULT 0;
    
    SELECT COALESCE(SUM(quantity_on_hand * unit_cost), 0) INTO total_value
    FROM inventory
    WHERE warehouse_id = warehouse_id_param;
    
    RETURN total_value;
END //
DELIMITER ;

-- Update customer tiers based on purchase totals
CALL UpdateCustomerTier(1);
CALL UpdateCustomerTier(4);
CALL UpdateCustomerTier(9);

-- Create final summary information
SELECT 'MySQL Test Database Initialization Complete' as status;
SELECT 
    (SELECT COUNT(*) FROM legacy_customers) as total_customers,
    (SELECT COUNT(*) FROM warehouses WHERE is_active = TRUE) as active_warehouses,
    (SELECT COUNT(*) FROM suppliers WHERE is_active = TRUE) as active_suppliers,
    (SELECT COUNT(*) FROM inventory) as inventory_records,
    (SELECT COUNT(*) FROM product_suppliers) as supplier_relationships;