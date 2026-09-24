from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel
from pathlib import Path
from datetime import datetime
import sqlite3


# =========================================================
# PATHS
# =========================================================

BASE_DIR = Path(__file__).resolve().parent.parent

DATA_DIR = BASE_DIR / "data"
DATA_DIR.mkdir(exist_ok=True)

DB_PATH = DATA_DIR / "retail_edge_ai.db"

FRONTEND_FILE = BASE_DIR / "frontend" / "index.html"


# =========================================================
# FASTAPI
# =========================================================

app = FastAPI(
    title="RetailEdge AI",
   version="1.0.0"
) 


# =========================================================
# DATABASE
# =========================================================

def get_db():

    connection = sqlite3.connect(DB_PATH)

    connection.row_factory = sqlite3.Row

    return connection

def add_column_if_missing(
    connection,
    table_name,
    column_name,
    column_definition
):
    columns = connection.execute(
        f"PRAGMA table_info({table_name})"
    ).fetchall()

    existing_columns = [row["name"] for row in columns]

    if column_name not in existing_columns:
        connection.execute(
            f"""
            ALTER TABLE {table_name}
            ADD COLUMN {column_name} {column_definition}
            """
        )

def initialize_database():

    connection = get_db()

    cursor = connection.cursor()

    # -----------------------------------------------------
    # PRODUCTS
    # -----------------------------------------------------

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS products (

            barcode TEXT PRIMARY KEY,

            sku TEXT UNIQUE NOT NULL,

            name TEXT NOT NULL,

            category TEXT NOT NULL,

            price REAL NOT NULL,

            shelf_id TEXT NOT NULL,

            stock INTEGER NOT NULL,

            threshold INTEGER NOT NULL

        )
    """)

    # -----------------------------------------------------
    # TROLLEYS
    # -----------------------------------------------------

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS trolleys (

            trolley_id TEXT PRIMARY KEY,

            status TEXT NOT NULL,

            weight_g REAL DEFAULT 0

        )
    """)

    # -----------------------------------------------------
    # CART
    # -----------------------------------------------------

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS cart_items (

            trolley_id TEXT NOT NULL,

            barcode TEXT NOT NULL,

            quantity INTEGER NOT NULL,

            PRIMARY KEY (
                trolley_id,
                barcode
            )

        )
    """)

    # -----------------------------------------------------
    # ALERTS
    # -----------------------------------------------------

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS alerts (

            id INTEGER PRIMARY KEY AUTOINCREMENT,

            alert_type TEXT NOT NULL,

            message TEXT NOT NULL,

            created_at TEXT NOT NULL

        )
    """)

    # -----------------------------------------------------
    # SHOPPER / QUEUE METRICS
    # -----------------------------------------------------

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS store_metrics (

            id INTEGER PRIMARY KEY CHECK(id = 1),

            customers_inside INTEGER DEFAULT 0,

            zone_a INTEGER DEFAULT 0,

            zone_b INTEGER DEFAULT 0,

            zone_c INTEGER DEFAULT 0,

            queue_people INTEGER DEFAULT 0,

            active_counters INTEGER DEFAULT 1,

            avg_service_seconds REAL DEFAULT 60

        )
    """)

    # -----------------------------------------------------
    # PRODUCT DATA
    # -----------------------------------------------------

    products = [

        (
            "890100000001",
            "SKU_CHIPS",
            "Potato Chips",
            "Snacks",
            30.0,
            "SHELF_A",
            20,
            5
        ),

        (
            "890100000002",
            "SKU_BISCUIT",
            "Biscuits",
            "Snacks",
            20.0,
            "SHELF_A",
            15,
            5
        ),

        (
            "890100000003",
            "SKU_JUICE",
            "Sprite",
            "Beverages",
            45.0,
            "SHELF_B",
            12,
            5
        ),

        (
            "890100000004",
            "SKU_SOAP",
            "Bath Soap",
            "Personal Care",
            35.0,
            "SHELF_C",
            10,
            4
        ),

        (
            "890100000005",
            "SKU_MILK",
            "Milk Pack",
            "Dairy",
            28.0,
            "SHELF_B",
            18,
            5
        ),

        (
            "890100000006",
            "SKU_COLA",
            "Cola Can",
            "Beverages",
            40.0,
            "SHELF_B",
            16,
            5
        )

    ]

    for product in products:

        cursor.execute("""
            INSERT OR IGNORE INTO products
            (
                barcode,
                sku,
                name,
                category,
                price,
                shelf_id,
                stock,
                threshold
            )

            VALUES (?, ?, ?, ?, ?, ?, ?, ?)

        """, product)

    # -----------------------------------------------------
    # TROLLEYS
    # -----------------------------------------------------

    trolley_ids = [
        "TROLLEY_01",
        "TROLLEY_02",
        "TROLLEY_03"
    ]

    for trolley_id in trolley_ids:

        cursor.execute("""
            INSERT OR IGNORE INTO trolleys
            (
                trolley_id,
                status,
                weight_g
            )

            VALUES (?, ?, ?)

        """, (
            trolley_id,
            "AVAILABLE",
            0
        ))

    # -----------------------------------------------------
    # METRICS
    # -----------------------------------------------------

    cursor.execute("""
        INSERT OR IGNORE INTO store_metrics
        (
            id
        )

        VALUES (1)
    """)

    add_column_if_missing(
        connection,
        "store_metrics",
        "average_dwell_seconds",
        "REAL DEFAULT 0"
    )

    add_column_if_missing(
        connection,
        "trolleys",
        "weight_g",
        "REAL DEFAULT 0"
    )

    connection.execute("""
    CREATE TABLE IF NOT EXISTS rfid_tags (
        rfid_uid TEXT PRIMARY KEY,
        barcode TEXT NOT NULL,
        created_at TEXT NOT NULL,
        FOREIGN KEY (barcode)
        REFERENCES products(barcode)
    )
""")

    connection.execute("""
        CREATE TABLE IF NOT EXISTS events (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            event_type TEXT NOT NULL,
            trolley_id TEXT,
            barcode TEXT,
            rfid_uid TEXT,
            value REAL,
            message TEXT,
            created_at TEXT NOT NULL
        )
    """)

    connection.execute("""
        CREATE TABLE IF NOT EXISTS sales (
            sale_id INTEGER PRIMARY KEY AUTOINCREMENT,
            trolley_id TEXT NOT NULL,
            total REAL NOT NULL,
            created_at TEXT NOT NULL
        )
    """)

    connection.execute("""
        CREATE TABLE IF NOT EXISTS sale_items (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            sale_id INTEGER NOT NULL,
            barcode TEXT NOT NULL,
            quantity INTEGER NOT NULL,
            price REAL NOT NULL,
            FOREIGN KEY (sale_id)
            REFERENCES sales(sale_id)
        )
    """)

    connection.commit()
    connection.close()
# =========================================================
# MODELS
# =========================================================

class ScanRequest(BaseModel):

    trolley_id: str

    barcode: str


class RemoveRequest(BaseModel):

    trolley_id: str

    barcode: str


class ReturnRequest(BaseModel):

    trolley_id: str

    barcode: str

    detected_shelf: str


class ShopperRequest(BaseModel):
    customers_inside: int
    zone_a: int
    zone_b: int
    zone_c: int
    average_dwell_seconds: float = 0


class QueueRequest(BaseModel):

    people: int

    active_counters: int

    average_service_seconds: float


class WeightRequest(BaseModel):

    trolley_id: str

    weight_g: float

class ProductMismatchRequest(BaseModel):

    trolley_id: str

    expected_barcode: str

    detected_barcode: str

    confidence: float
# =========================================================
# HELPERS
# =========================================================

def current_time():

    return datetime.now().isoformat(
        timespec="seconds"
    )


def add_alert(
    cursor,
    alert_type,
    message
):

    cursor.execute("""
        INSERT INTO alerts
        (
            alert_type,
            message,
            created_at
        )

        VALUES (?, ?, ?)

    """, (
        alert_type,
        message,
        current_time()
    ))


def get_trolley_cart(trolley_id):

    connection = get_db()

    rows = connection.execute("""
        SELECT

            cart_items.barcode,

            cart_items.quantity,

            products.name,

            products.price,

            products.category

        FROM cart_items

        JOIN products

        ON cart_items.barcode =
           products.barcode

        WHERE cart_items.trolley_id = ?

        ORDER BY products.name

    """, (
        trolley_id,
    )).fetchall()

    trolley = connection.execute("""
        SELECT *

        FROM trolleys

        WHERE trolley_id = ?

    """, (
        trolley_id,
    )).fetchone()

    connection.close()

    if trolley is None:

        raise HTTPException(
            status_code=404,
            detail="Trolley not found"
        )

    items = []

    total = 0

    item_count = 0

    for row in rows:

        line_total = (
            row["price"]
            *
            row["quantity"]
        )

        total += line_total

        item_count += row["quantity"]

        items.append({

            "barcode":
                row["barcode"],

            "name":
                row["name"],

            "category":
                row["category"],

            "quantity":
                row["quantity"],

            "price":
                row["price"],

            "line_total":
                line_total

        })

    return {

        "trolley_id":
            trolley_id,

        "status":
            trolley["status"],

        "weight_g":
            trolley["weight_g"],

        "items":
            items,

        "item_count":
            item_count,

        "total":
            round(total, 2)

    }


# =========================================================
# WEBSITE
# =========================================================

@app.get("/")
def home():

    if not FRONTEND_FILE.exists():

        return {
            "error":
                "frontend/index.html not found"
        }

    return FileResponse(
        FRONTEND_FILE
    )


# =========================================================
# HEALTH
# =========================================================

@app.get("/api/health")
def health():

    return {

        "status":
            "running",

        "project":
            "RetailEdge AI"

    }


# =========================================================
# PRODUCTS
# =========================================================

@app.get("/api/products")
def get_products():

    connection = get_db()

    rows = connection.execute("""
        SELECT *

        FROM products

        ORDER BY name
    """).fetchall()

    connection.close()

    return [
        dict(row)
        for row in rows
    ]


# =========================================================
# TROLLEY
# =========================================================

@app.get("/api/trolley/{trolley_id}")
def trolley(
    trolley_id: str
):

    return get_trolley_cart(
        trolley_id
    )


# =========================================================
# SCAN PRODUCT
# =========================================================

@app.post("/api/scan")
def scan_product(
    request: ScanRequest
):

    connection = get_db()

    cursor = connection.cursor()

    # Find product
    product = cursor.execute("""
        SELECT *

        FROM products

        WHERE barcode = ?

    """, (
        request.barcode,
    )).fetchone()

    if product is None:

        connection.close()

        raise HTTPException(
            status_code=404,
            detail="Barcode not registered"
        )

    # Check trolley
    trolley = cursor.execute("""
        SELECT *

        FROM trolleys

        WHERE trolley_id = ?

    """, (
        request.trolley_id,
    )).fetchone()

    if trolley is None:

        connection.close()

        raise HTTPException(
            status_code=404,
            detail="Trolley not found"
        )

    # Check stock
    if product["stock"] <= 0:

        connection.close()

        raise HTTPException(
            status_code=400,
            detail=f"{product['name']} is out of stock"
        )

    # Reduce stock
    cursor.execute("""
        UPDATE products

        SET stock = stock - 1

        WHERE barcode = ?

    """, (
        request.barcode,
    ))

    # Check existing cart quantity
    existing = cursor.execute("""
        SELECT quantity

        FROM cart_items

        WHERE trolley_id = ?

        AND barcode = ?

    """, (
        request.trolley_id,
        request.barcode
    )).fetchone()

    if existing:

        cursor.execute("""
            UPDATE cart_items

            SET quantity =
                quantity + 1

            WHERE trolley_id = ?

            AND barcode = ?

        """, (
            request.trolley_id,
            request.barcode
        ))

    else:

        cursor.execute("""
            INSERT INTO cart_items
            (
                trolley_id,
                barcode,
                quantity
            )

            VALUES (?, ?, 1)

        """, (
            request.trolley_id,
            request.barcode
        ))

    # Trolley becomes active
    cursor.execute("""
        UPDATE trolleys

        SET status = 'SHOPPING'

        WHERE trolley_id = ?

    """, (
        request.trolley_id,
    ))

    # Low stock
    new_stock = product["stock"] - 1

    if new_stock <= product["threshold"]:

        add_alert(
            cursor,
            "LOW_STOCK",
            (
                f"{product['name']} is low on stock. "
                f"Remaining: {new_stock}"
            )
        )

    connection.commit()

    connection.close()

    return {

        "success":
            True,

        "product": {

            "barcode":
                product["barcode"],

            "name":
                product["name"],

            "price":
                product["price"],

            "category":
                product["category"]

        },

        "cart":
            get_trolley_cart(
                request.trolley_id
            )

    }


# =========================================================
# REMOVE ITEM
# =========================================================

@app.post("/api/remove")
def remove_item(
    request: RemoveRequest
):

    connection = get_db()

    cursor = connection.cursor()

    item = cursor.execute("""
        SELECT quantity

        FROM cart_items

        WHERE trolley_id = ?

        AND barcode = ?

    """, (
        request.trolley_id,
        request.barcode
    )).fetchone()

    if item is None:

        connection.close()

        raise HTTPException(
            status_code=404,
            detail="Item is not in trolley"
        )

    if item["quantity"] > 1:

        cursor.execute("""
            UPDATE cart_items

            SET quantity =
                quantity - 1

            WHERE trolley_id = ?

            AND barcode = ?

        """, (
            request.trolley_id,
            request.barcode
        ))

    else:

        cursor.execute("""
            DELETE FROM cart_items

            WHERE trolley_id = ?

            AND barcode = ?

        """, (
            request.trolley_id,
            request.barcode
        ))

    # Product becomes available again
    cursor.execute("""
        UPDATE products

        SET stock = stock + 1

        WHERE barcode = ?

    """, (
        request.barcode,
    ))

    connection.commit()

    connection.close()

    return get_trolley_cart(
        request.trolley_id
    )


# =========================================================
# RETURN TO SHELF
# =========================================================

@app.post("/api/return")
def return_product(
    request: ReturnRequest
):

    connection = get_db()

    cursor = connection.cursor()

    product = cursor.execute("""
        SELECT *

        FROM products

        WHERE barcode = ?

    """, (
        request.barcode,
    )).fetchone()

    if product is None:

        connection.close()

        raise HTTPException(
            status_code=404,
            detail="Product not found"
        )

    expected_shelf = product["shelf_id"]

    if (
        request.detected_shelf
        !=
        expected_shelf
    ):

        add_alert(
            cursor,
            "MISPLACED_PRODUCT",
            (
                f"{product['name']} detected on "
                f"{request.detected_shelf}. "
                f"Expected {expected_shelf}."
            )
        )

        result = "MISPLACED"

    else:

        result = "CORRECT"

    connection.commit()

    connection.close()

    return {

        "success":
            True,

        "product":
            product["name"],

        "expected_shelf":
            expected_shelf,

        "detected_shelf":
            request.detected_shelf,

        "result":
            result

    }

# =========================================================
# PRODUCT MISMATCH
# =========================================================

@app.post("/api/product-mismatch")
def product_mismatch(
    request: ProductMismatchRequest
):

    connection = get_db()

    cursor = connection.cursor()

    expected = cursor.execute(
        """
        SELECT name
        FROM products
        WHERE barcode = ?
        """,
        (
            request.expected_barcode,
        )
    ).fetchone()

    detected = cursor.execute(
        """
        SELECT name
        FROM products
        WHERE barcode = ?
        """,
        (
            request.detected_barcode,
        )
    ).fetchone()

    expected_name = (
        expected["name"]
        if expected
        else request.expected_barcode
    )

    detected_name = (
        detected["name"]
        if detected
        else request.detected_barcode
    )

    message = (
        f"Trolley {request.trolley_id}: "
        f"barcode indicates {expected_name}, "
        f"camera detected {detected_name} "
        f"(confidence {request.confidence:.2f})."
    )

    cursor.execute(
        """
        INSERT INTO alerts
        (
            alert_type,
            message,
            created_at
        )
        VALUES (?, ?, ?)
        """,
        (
            "PRODUCT_MISMATCH",
            message,
            datetime.now().isoformat(
                timespec="seconds"
            )
        )
    )

    connection.commit()

    connection.close()

    return {

        "success":
            True,

        "alert":
            "PRODUCT_MISMATCH",

        "message":
            message

    }
# =========================================================
# CHECKOUT
# =========================================================

@app.post("/api/checkout/{trolley_id}")
def checkout(
    trolley_id: str
):

    connection = get_db()

    cursor = connection.cursor()

    trolley = cursor.execute("""
        SELECT *

        FROM trolleys

        WHERE trolley_id = ?

    """, (
        trolley_id,
    )).fetchone()

    if trolley is None:

        connection.close()

        raise HTTPException(
            status_code=404,
            detail="Trolley not found"
        )

    cart = get_trolley_cart(
        trolley_id
    )

    # Current cart value becomes the sale value.
    # For this software prototype we clear
    # the shopping session after checkout.

    cursor.execute("""
        DELETE FROM cart_items

        WHERE trolley_id = ?

    """, (
        trolley_id,
    ))

    cursor.execute("""
        UPDATE trolleys

        SET status = 'AVAILABLE',

            weight_g = 0

        WHERE trolley_id = ?

    """, (
        trolley_id,
    ))

    connection.commit()

    connection.close()

    return {

        "success":
            True,

        "message":
            "Checkout completed",

        "total":
            cart["total"]

    }


# =========================================================
# WEIGHT SENSOR
# =========================================================

@app.post("/api/weight")
def update_weight(
    request: WeightRequest
):

    connection = get_db()

    cursor = connection.cursor()

    exists = cursor.execute("""
        SELECT trolley_id

        FROM trolleys

        WHERE trolley_id = ?

    """, (
        request.trolley_id,
    )).fetchone()

    if exists is None:

        connection.close()

        raise HTTPException(
            status_code=404,
            detail="Trolley not found"
        )

    cursor.execute("""
        UPDATE trolleys

        SET weight_g = ?

        WHERE trolley_id = ?

    """, (
        request.weight_g,
        request.trolley_id
    ))

    connection.commit()

    connection.close()

    return get_trolley_cart(
        request.trolley_id
    )


# =========================================================
# SHOPPER ANALYTICS
# =========================================================

@app.post("/api/shopper")
def shopper_metrics(
    request: ShopperRequest
):

    connection = get_db()

    connection.execute("""
        UPDATE store_metrics

        SET customers_inside = ?,
            zone_a = ?,
            zone_b = ?,
            zone_c = ?,
            average_dwell_seconds = ?

        WHERE id = 1

    """, (
        request.customers_inside,
        request.zone_a,
        request.zone_b,
        request.zone_c,
        request.average_dwell_seconds
    ))

    connection.commit()

    connection.close()

    return {
        "success": True,
        "customers_inside": request.customers_inside,
        "zone_a": request.zone_a,
        "zone_b": request.zone_b,
        "zone_c": request.zone_c,
        "average_dwell_seconds": request.average_dwell_seconds
    }


# =========================================================
# QUEUE
# =========================================================

@app.post("/api/queue")
def queue_metrics(
    request: QueueRequest
):

    connection = get_db()

    connection.execute("""
        UPDATE store_metrics

        SET queue_people = ?,

            active_counters = ?,

            avg_service_seconds = ?

        WHERE id = 1

    """, (
        request.people,
        max(1, request.active_counters),
        request.average_service_seconds
    ))

    connection.commit()

    connection.close()

    return {
        "success": True
    }


# =========================================================
# SHELF AI EVENT
# =========================================================

@app.post("/api/shelf-event")
def shelf_event(
    request: ReturnRequest
):

    connection = get_db()

    cursor = connection.cursor()

    product = cursor.execute("""
        SELECT *

        FROM products

        WHERE sku = ?

    """, (
        request.barcode,
    )).fetchone()

    # If SKU isn't found, also allow barcode input
    if product is None:

        product = cursor.execute("""
            SELECT *

            FROM products

            WHERE barcode = ?

        """, (
            request.barcode,
        )).fetchone()

    if product is None:

        connection.close()

        raise HTTPException(
            status_code=404,
            detail="Product/SKU not found"
        )

    if (
        request.detected_shelf
        !=
        product["shelf_id"]
    ):

        add_alert(
            cursor,
            "MISPLACED_PRODUCT",
            (
                f"{product['name']} detected on "
                f"{request.detected_shelf}. "
                f"Expected {product['shelf_id']}."
            )
        )

        result = "MISPLACED"

    else:

        result = "CORRECT"

    connection.commit()

    connection.close()

    return {

        "success": True,

        "product":
            product["name"],

        "expected_shelf":
            product["shelf_id"],

        "detected_shelf":
            request.detected_shelf,

        "result":
            result

    }


# =========================================================
# DASHBOARD
# =========================================================

@app.get("/api/dashboard")
def dashboard():

    connection = get_db()

    products = connection.execute("""
        SELECT *

        FROM products

        ORDER BY name

    """).fetchall()

    trolleys = connection.execute("""
        SELECT *

        FROM trolleys

        ORDER BY trolley_id

    """).fetchall()

    alerts = connection.execute("""
        SELECT *

        FROM alerts

        ORDER BY id DESC

        LIMIT 20

    """).fetchall()

    metrics = connection.execute("""
        SELECT *

        FROM store_metrics

        WHERE id = 1

    """).fetchone()

    connection.close()

    inventory = []

    low_stock_count = 0

    total_inventory = 0

    for product in products:

        low = (
            product["stock"]
            <=
            product["threshold"]
        )

        if low:
            low_stock_count += 1

        total_inventory += product["stock"]

        inventory.append({

            "barcode":
                product["barcode"],

            "sku":
                product["sku"],

            "name":
                product["name"],

            "category":
                product["category"],

            "price":
                product["price"],

            "shelf":
                product["shelf_id"],

            "stock":
                product["stock"],

            "threshold":
                product["threshold"],

            "low_stock":
                low,

            "status":
                "LOW STOCK"
                if low
                else "OK"

        })

    trolley_data = []

    active_trolleys = 0

    for trolley in trolleys:

        state = get_trolley_cart(
            trolley["trolley_id"]
        )

        if trolley["status"] == "SHOPPING":

            active_trolleys += 1

        trolley_data.append({

            "trolley_id":
                trolley["trolley_id"],

            "status":
                trolley["status"],

            "items":
                state["item_count"],

            "total":
                state["total"],

            "weight_g":
                trolley["weight_g"]

        })

    queue_people = metrics["queue_people"]

    counters = max(
        1,
        metrics["active_counters"]
    )

    wait_minutes = (

        queue_people
        *
        metrics["avg_service_seconds"]

        /

        counters

        /

        60

    )

    return {

        "customers_inside":
            metrics["customers_inside"],

        "active_trolleys":
            active_trolleys,

        "low_stock":
            low_stock_count,

        "total_inventory":
            total_inventory,

        "average_dwell_seconds":
            metrics["average_dwell_seconds"],   

        "zones": {

            "zone_a":
                metrics["zone_a"],

            "zone_b":
                metrics["zone_b"],

            "zone_c":
                metrics["zone_c"]

        },
        "average_dwell_seconds":
            metrics["average_dwell_seconds"],

        "queue": {

            "people":
                queue_people,

            "counters":
                counters,

            "average_service_seconds":
                metrics["avg_service_seconds"],

            "wait_minutes":
                round(
                    wait_minutes,
                    1
                )

        },

        "inventory":
            inventory,

        "trolleys":
            trolley_data,

        "alerts": [

            {

                "type":
                    row["alert_type"],

                "message":
                    row["message"],

                "created_at":
                    row["created_at"]

            }

            for row in alerts

        ]

    }


# =========================================================
# RESET
# =========================================================

@app.post("/api/reset")
def reset_demo():

    connection = get_db()

    cursor = connection.cursor()

    cursor.execute(
        "DELETE FROM cart_items"
    )

    cursor.execute(
        "DELETE FROM alerts"
    )

    stock_values = {

        "890100000001": 20,

        "890100000002": 15,

        "890100000003": 12,

        "890100000004": 10,

        "890100000005": 18,

        "890100000006": 16

    }

    for barcode, stock in stock_values.items():

        cursor.execute("""
            UPDATE products

            SET stock = ?

            WHERE barcode = ?

        """, (
            stock,
            barcode
        ))

    cursor.execute("""
        UPDATE trolleys

        SET status = 'AVAILABLE',

            weight_g = 0
    """)

    cursor.execute("""
        UPDATE store_metrics

        SET customers_inside = 0,

            zone_a = 0,

            zone_b = 0,

            zone_c = 0,

            queue_people = 0,

            active_counters = 1,

            avg_service_seconds = 60

        WHERE id = 1
    """)

    connection.commit()

    connection.close()

    return {

        "success":
            True,

        "message":
            "Demo reset successfully"

    }


# =========================================================
# STARTUP
# =========================================================

@app.on_event("startup")
def startup_event():

    initialize_database()