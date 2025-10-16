import json
import pandas as pd
import polars as pl

json_name = "added_data.json"

with open(json_name, "r", encoding="utf-8") as file:
    data = json.load(file)

df = pl.read_csv("./data.csv", separator=";").with_columns()
df_copy = df.clone()
for key,value in data.items():
    row = df.filter(pl.col("answer_pattern") == key)
    for quesion in value:
        row_clone = row.clone()
        row_clone = row.with_columns(
            pl.lit(quesion).alias("text")  # Add the question as the 'text' column
        )
        df_copy = df_copy.vstack(row_clone)

df_copy.write_csv("./data_extended.csv", separator=";")