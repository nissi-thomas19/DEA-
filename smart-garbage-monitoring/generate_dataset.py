import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import random

np.random.seed(42)

num_records = 50000
bins = 100

data = []

start_time = datetime.now()

for i in range(num_records):
    bin_id = random.randint(1, bins)
    timestamp = start_time + timedelta(minutes=i)

    fill_level = np.random.uniform(0, 100)
    temperature = np.random.uniform(20, 50)
    humidity = np.random.uniform(30, 90)

    if fill_level < 30:
        status = "Empty"
    elif fill_level < 75:
        status = "Half-Filled"
    else:
        status = "Full"

    data.append([bin_id, timestamp, fill_level, temperature, humidity, status])

df = pd.DataFrame(data, columns=[
    "Bin_ID", "Timestamp", "Fill_Level",
    "Temperature", "Humidity", "Status"
])

df.to_csv("data/garbage_large_dataset.csv", index=False)

print("Large dataset generated successfully!")
