import pandas as pd
import matplotlib.pyplot as plt


df = pd.read_csv("final_india_dataset.csv")

print(f"DATASET REPORT")
print(f"-----------------")
print(f"Total Fires: {len(df)}")
print(f"Latitudes:  {df['latitude'].min()} to {df['latitude'].max()}")
print(f"Longitudes: {df['longitude'].min()} to {df['longitude'].max()}")

north_fires = len(df[df['latitude'] > 28])
south_fires = len(df[df['latitude'] < 20])
print(f"-----------------")
print(f"North India Fires: {north_fires}")
print(f"South India Fires: {south_fires}")
print(f"-----------------")

if north_fires > 100 and south_fires > 100:
    print("both North and South .")
else:
    print("Not All")