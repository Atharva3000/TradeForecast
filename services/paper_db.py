import sqlite3
import logging
from services.db import get_db_connection

logger = logging.getLogger(__name__)

def get_portfolio(username: str) -> dict:
    """
    Get the user's paper trading portfolio: cash balance and open positions.
    If the portfolio does not exist, initialize it with default capital (10 Lakhs).
    """
    username = username.strip().lower()
    conn = get_db_connection()
    cursor = conn.cursor()
    
    try:
        # Check cash balance
        cursor.execute("SELECT cash_balance, currency FROM paper_portfolio WHERE username = ?", (username,))
        row = cursor.fetchone()
        
        if not row:
            # Initialize with default 1,000,000.0 (10 Lakhs INR)
            # Use currency based on user's selected country if possible
            cursor.execute("SELECT country, investment_capital FROM users WHERE username = ?", (username,))
            user_row = cursor.fetchone()
            
            initial_cash = 1000000.0
            currency = "₹"
            if user_row:
                if user_row["investment_capital"] and user_row["investment_capital"] > 0:
                    initial_cash = float(user_row["investment_capital"])
                
                country = (user_row["country"] or "").strip().lower()
                # US and other global markets default to $
                if country in ["us", "usa", "united states", "uk", "canada", "global"]:
                    currency = "$"
            
            cursor.execute(
                "INSERT INTO paper_portfolio (username, cash_balance, currency) VALUES (?, ?, ?)",
                (username, initial_cash, currency)
            )
            conn.commit()
            cash_balance = initial_cash
        else:
            cash_balance = row["cash_balance"]
            currency = row["currency"]
            
        # Get open positions
        cursor.execute(
            "SELECT ticker, average_price, quantity FROM paper_positions WHERE username = ? AND quantity > 0",
            (username,)
        )
        positions = [dict(r) for r in cursor.fetchall()]
        
        return {
            "username": username,
            "cash_balance": cash_balance,
            "currency": currency,
            "positions": positions
        }
    finally:
        conn.close()

def execute_order(username: str, ticker: str, direction: str, quantity: float, price: float, order_type: str = "Market") -> dict:
    """
    Executes a paper trading order, updating balances, positions, and history.
    """
    username = username.strip().lower()
    ticker = ticker.strip().upper()
    direction = direction.strip().upper()
    order_type = order_type.strip()
    
    if quantity <= 0:
        raise ValueError("Quantity must be greater than zero.")
    if price <= 0:
        raise ValueError("Price must be greater than zero.")
        
    conn = get_db_connection()
    cursor = conn.cursor()
    
    try:
        # Get portfolio
        cursor.execute("SELECT cash_balance, currency FROM paper_portfolio WHERE username = ?", (username,))
        port_row = cursor.fetchone()
        if not port_row:
            conn.close()
            # Initialize portfolio first
            portfolio = get_portfolio(username)
            conn = get_db_connection()
            cursor = conn.cursor()
            cash_balance = portfolio["cash_balance"]
            currency = portfolio["currency"]
        else:
            cash_balance = port_row["cash_balance"]
            currency = port_row["currency"]
            
        # Get current position
        cursor.execute(
            "SELECT average_price, quantity FROM paper_positions WHERE username = ? AND ticker = ?",
            (username, ticker)
        )
        pos_row = cursor.fetchone()
        current_qty = pos_row["quantity"] if pos_row else 0.0
        current_avg = pos_row["average_price"] if pos_row else 0.0
        
        total_value = quantity * price
        
        if direction == "BUY":
            # Check cash
            if cash_balance < total_value:
                raise ValueError(f"Insufficient virtual capital. Order requires {currency}{total_value:,.2f}, but you have {currency}{cash_balance:,.2f}.")
                
            new_cash = cash_balance - total_value
            new_qty = current_qty + quantity
            new_avg = ((current_qty * current_avg) + total_value) / new_qty
            
            # Update cash
            cursor.execute(
                "UPDATE paper_portfolio SET cash_balance = ? WHERE username = ?",
                (new_cash, username)
            )
            
            # Update position
            if pos_row:
                cursor.execute(
                    "UPDATE paper_positions SET average_price = ?, quantity = ? WHERE username = ? AND ticker = ?",
                    (new_avg, new_qty, username, ticker)
                )
            else:
                cursor.execute(
                    "INSERT INTO paper_positions (username, ticker, average_price, quantity) VALUES (?, ?, ?, ?)",
                    (username, ticker, new_avg, new_qty)
                )
                
        elif direction == "SELL":
            # Check position
            if current_qty < quantity:
                raise ValueError(f"Insufficient shares. You hold {current_qty} of {ticker}, but tried to sell {quantity}.")
                
            new_cash = cash_balance + total_value
            new_qty = current_qty - quantity
            
            # Update cash
            cursor.execute(
                "UPDATE paper_portfolio SET cash_balance = ? WHERE username = ?",
                (new_cash, username)
            )
            
            # Update position
            if new_qty <= 0.0001:  # float margin
                cursor.execute(
                    "DELETE FROM paper_positions WHERE username = ? AND ticker = ?",
                    (username, ticker)
                )
            else:
                cursor.execute(
                    "UPDATE paper_positions SET quantity = ? WHERE username = ? AND ticker = ?",
                    (new_qty, username, ticker)
                )
        else:
            raise ValueError(f"Invalid direction '{direction}'. Must be BUY or SELL.")
            
        # Log order
        cursor.execute(
            """
            INSERT INTO paper_orders (username, ticker, direction, order_type, quantity, price, status)
            VALUES (?, ?, ?, ?, ?, ?, 'FILLED')
            """,
            (username, ticker, direction, order_type, quantity, price)
        )
        
        conn.commit()
        verb = "bought" if direction == "BUY" else "sold"
        return {
            "status": "success",
            "message": f"Successfully {verb} {quantity} shares of {ticker} at {currency}{price:,.2f}."
        }
    except Exception as e:
        conn.rollback()
        logger.error("Error executing paper trade for %s: %s", username, e)
        raise e
    finally:
        conn.close()

def reset_portfolio(username: str) -> dict:
    """
    Reset user's virtual cash balance to default (from profile or 10L) and delete all trades & positions.
    """
    username = username.strip().lower()
    conn = get_db_connection()
    cursor = conn.cursor()
    
    try:
        # Determine initial cash capital based on user's settings
        cursor.execute("SELECT investment_capital, country FROM users WHERE username = ?", (username,))
        user_row = cursor.fetchone()
        
        initial_cash = 1000000.0
        currency = "₹"
        if user_row:
            if user_row["investment_capital"] and user_row["investment_capital"] > 0:
                initial_cash = float(user_row["investment_capital"])
            
            country = (user_row["country"] or "").strip().lower()
            if country in ["us", "usa", "united states", "uk", "canada", "global"]:
                currency = "$"
                
        # Delete positions & orders
        cursor.execute("DELETE FROM paper_positions WHERE username = ?", (username,))
        cursor.execute("DELETE FROM paper_orders WHERE username = ?", (username,))
        
        # Reset balance
        cursor.execute(
            "INSERT OR REPLACE INTO paper_portfolio (username, cash_balance, currency) VALUES (?, ?, ?)",
            (username, initial_cash, currency)
        )
        conn.commit()
        return {"status": "success", "message": "Virtual trading portfolio reset successfully."}
    except Exception as e:
        conn.rollback()
        logger.error("Error resetting portfolio for %s: %s", username, e)
        raise e
    finally:
        conn.close()

def get_order_history(username: str) -> list[dict]:
    """
    Get list of all orders executed by user.
    """
    username = username.strip().lower()
    conn = get_db_connection()
    cursor = conn.cursor()
    
    try:
        cursor.execute(
            "SELECT ticker, direction, order_type, quantity, price, status, created_at FROM paper_orders WHERE username = ? ORDER BY created_at DESC",
            (username,)
        )
        return [dict(r) for r in cursor.fetchall()]
    finally:
        conn.close()
