import pandas as pd
import sqlite3

# Load CSV files into pandas DataFrames
experiment1_df = pd.read_csv("../data/experiment_vol1.csv")
experiment2_df = pd.read_csv("../data/experiment_vol2.csv")
experiment3_df = pd.read_csv("../data/experiment_vol3.csv")

# Create a connection to the SQLite database
conn = sqlite3.connect('../data/experiments.db')

# Write the DataFrames to the SQLite database
experiment1_df.to_sql('experiment1', conn, if_exists='replace', index=False)
experiment2_df.to_sql('experiment2', conn, if_exists='replace', index=False)
experiment3_df.to_sql('experiment3', conn, if_exists='replace', index=False)

# Define SQL queries with the correct column names
queries = {
    'total_propellers_thrust_gt_12': """
        SELECT COUNT(*) AS total_propellers
        FROM experiment1
        WHERE "Thrust Coefficient Output" > 0.12;
    """,
    'reorder_by_efficiency_desc': """
        SELECT *
        FROM experiment1
        ORDER BY "Efficiency Output" DESC;
    """,
    'least_100_performing_propellers': """
        SELECT *
        FROM experiment1
        ORDER BY "Power Coefficient Output" ASC
        LIMIT 100;
    """,
    'total_negative_or_zero_efficiency': """
        SELECT COUNT(*) AS total_propellers
        FROM (
            SELECT * FROM experiment1
            UNION ALL
            SELECT * FROM experiment2
            UNION ALL
            SELECT * FROM experiment3
        ) AS merged_experiments
        WHERE "Efficiency Output" <= 0;
    """
}

# Execute SQL queries and print results
with conn:
    for query_name, query in queries.items():
        result = pd.read_sql_query(query, conn)
        print(f"Result for {query_name}:")
        print(result)
        print("\n")

# Close the connection
conn.close()
