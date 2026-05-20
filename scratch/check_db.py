import sqlite3
conn = sqlite3.connect('predictions.sqlite3')
cursor = conn.execute("PRAGMA table_info(predictions)")
columns = [row[1] for row in cursor.fetchall()]
print(columns)
conn.close()
