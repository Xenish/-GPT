import matplotlib
matplotlib.use("Agg")  # Qt'ye ihtiyaç yok, direkt PNG üret

import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("ml_wf_metrics_15m.csv")

print("Columns:")
print(df.columns)

print("\nBasic stats:")
print(df[["precision", "recall", "accuracy", "f1"]].describe())

print("\nLowest precision blocks:")
print(df.sort_values("precision").head(10)[["block_id", "precision", "recall", "pred_pos_rate"]])

print("\nHighest precision blocks:")
print(df.sort_values("precision", ascending=False).head(10)[["block_id", "precision", "recall", "pred_pos_rate"]])

# grafik
df.plot(x="block_id", y=["precision", "recall", "accuracy", "f1"])
plt.title("Walk-forward metrics (15m)")
plt.xlabel("block_id")
plt.ylabel("score")
plt.tight_layout()
plt.savefig("wf_metrics_15m.png")

print("\nGrafik 'wf_metrics_15m.png' olarak kaydedildi.")
