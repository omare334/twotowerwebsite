# init_db.py
import os
import psycopg2
import pandas as pd

# Load the CSV data
csv_path = 'results_negative.csv'  # Replace with the path to your CSV file
data = pd.read_csv(csv_path)

# Database connection
database_url = os.getenv("DATABASE_URL", "postgresql://user:password@localhost:5432/dbname")
conn = psycopg2.connect(database_url)
cur = conn.cursor()

# Create the table without a primary key
cur.execute("""
    CREATE TABLE IF NOT EXISTS passages (
        query TEXT,
        passage_text TEXT,
        negative_sample TEXT
    );
""")
conn.commit()

# Insert data from CSV into the database
for _, row in data.iterrows():
    cur.execute("""
        INSERT INTO passages (query, passage_text, negative_sample)
        VALUES (%s, %s, %s);
    """, (row['query'], row['passage_text'], row['negative_sample']))

conn.commit()

# Close the connection
cur.close()
conn.close()
print("Database initialized with CSV data.")
