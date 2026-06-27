from fastmcp import FastMCP
import sqlite3
import os
import uuid
import datetime
mcp = FastMCP()

def connect_db():
    db_path = os.path.join("D:\\Projects\\ambient_ai\\database", "finance.db")
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS transactions (
            transaction_id TEXT PRIMARY KEY,
            amount REAL NOT NULL,
            category TEXT NOT NULL,
            category_id INTEGER,
            account_id INTEGER,
            date TEXT NOT NULL,
            description TEXT,
            type TEXT NOT NULL CHECK(type IN ('income', 'expense'))
        );
    ''')
    
    cursor.execute('''
                   CREATE TABLE IF NOT EXISTS categories (
        category_id INTEGER PRIMARY KEY AUTOINCREMENT,
        name TEXT NOT NULL UNIQUE,
        type TEXT NOT NULL CHECK(type IN ('Income', 'Expense'))
    );
 ''')
    
    cursor.execute('''
                       CREATE TABLE IF NOT EXISTS accounts (
        account_id INTEGER PRIMARY KEY AUTOINCREMENT,
        name TEXT NOT NULL UNIQUE,
        type TEXT NOT NULL,
        initial_balance REAL NOT NULL DEFAULT 0
    );
    ''')
    
    conn.commit()
    return conn

@mcp.tool
def log_transaction(amount: float, category: str, date: str = None, description: str = None):
    """
    Log a financial expense.

    Args:
        transaction_id: (Optional) A unique identifier for the transaction. If not provided, a UUID will be generated.
        amount: (Required) The amount of the expense.
        category: (Required) The category of the expense (e.g., 'Food', 'Transport').
        date: (Optional) The date of the expense in 'YYYY-MM-DD' format. Defaults to today if not provided.
        description: (Required) A brief description of the expense.
    """
    if date is None:
        date = datetime.date.today().isoformat()
    
    conn = connect_db()
    cursor = conn.cursor()
    
    try:
        cursor.execute('''
            INSERT INTO transactions (transaction_id, amount, category, date, description, type)
            VALUES (?, ?, ?, ?, ?, 'expense')
        ''', (str(uuid.uuid4()), amount, category, date, description))
        conn.commit()
        return f"Expense logged: {amount} in category '{category}' on {date}."
    except Exception as e:
        return f"Error: {e}"
    
@mcp.tool
def view_transactions(fetch_type: str = "Many",limit: int = 10):
    """ 
    View recent transactions.
    Also to be used when no transaction_id is provided to update_transaction.
    Args:
        fetch_type: (Optional) Type of fetch: "One": to fetch one record, "All": to fetch all records, "Many": to fetch a limited number of records. Defaults to "Many".
        limit: (Optional) Number of recent transactions to retrieve. Defaults to 10.
    Returns:
        List of recent transactions.
    """
    conn = connect_db()
    cursor = conn.cursor()
    cursor.execute('SELECT * FROM transactions ORDER BY date DESC')
    if fetch_type == "One":
        transactions = cursor.fetchone()
    elif fetch_type == "All":
        transactions = cursor.fetchall()
    elif fetch_type == "Many":
        transactions = cursor.fetchmany(limit)
    output = ""
    for t in transactions:
        output += f"ID: {t[0]}, Amount: {t[1]}, Category: {t[2]}, Date: {t[5]}, Description: {t[6]}, Type: {t[7]}\n"
    return output

@mcp.tool
def update_transaction(transaction_id: str, amount: float = None, category: str = None, date: str = None, description: str = None):
    """
    Update an existing transaction.
    Args:
        transaction_id: (Required) The unique identifier of the transaction to update.
        amount: (Optional) New amount.
        category: (Optional) New category.
        date: (Optional) New date in 'YYYY-MM-DD' format.
        description: (Optional) New description.
    Returns:
        Success or error message.
    """
    conn = connect_db()
    cursor = conn.cursor()
    
    fields = []
    values = []
    
    if amount is not None:
        fields.append("amount = ?")
        values.append(amount)
    if category is not None:
        fields.append("category = ?")
        values.append(category)
    if date is not None:
        fields.append("date = ?")
        values.append(date)
    if description is not None:
        fields.append("description = ?")
        values.append(description)
    
    if not fields:
        return "No fields to update."
    
    values.append(transaction_id)
    sql = f"UPDATE transactions SET {', '.join(fields)} WHERE transaction_id = ?"
    
    try:
        cursor.execute(sql, values)
        conn.commit()
        if cursor.rowcount == 0:
            return "Transaction not found."
        return "Transaction updated successfully."
    except Exception as e:
        return f"Error: {e}"
# log_expense(15.75, 'Food', '2023-10-01', 'Lunch at cafe')