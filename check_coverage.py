import pandas as pd
import matplotlib.pyplot as plt

# Load your data
df = pd.read_csv("final_india_dataset.csv")

print(f"📊 DATASET REPORT")
print(f"-----------------")
print(f"Total Fires: {len(df)}")
# Check latitude spread (South is ~8, North is ~37)
print(f"Latitudes:  {df['latitude'].min()} to {df['latitude'].max()}")
# Check longitude spread (West is ~68, East is ~97)
print(f"Longitudes: {df['longitude'].min()} to {df['longitude'].max()}")

# Quick Text-Based Map (If you can't see the image)
north_fires = len(df[df['latitude'] > 28])
south_fires = len(df[df['latitude'] < 20])
print(f"-----------------")
print(f"North India Fires: {north_fires}")
print(f"South India Fires: {south_fires}")
print(f"-----------------")

if north_fires > 100 and south_fires > 100:
    print("✅ GREAT! You have data from both North and South India.")
else:
    print("⚠️ WARNING: Your data might be clustered in one region.")