import matplotlib

matplotlib.use("Agg")
import pandas as pd
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt

from utils import RESULTS_DIR

df = pd.read_csv(RESULTS_DIR / "finetuned_crosstab.csv", index_col=0)

class_names = df.index.tolist()
y_true = []
y_pred = []

for i, true_cls in enumerate(class_names):
    for j, pred_cls in enumerate(df.columns):
        count = df.iloc[i, j]
        
        y_true.extend([true_cls] * count)
        y_pred.extend([pred_cls] * count)

conf_matr = confusion_matrix(y_true, y_pred, labels=class_names)
disp = ConfusionMatrixDisplay(confusion_matrix=conf_matr, display_labels=class_names)
fig, ax = plt.subplots(figsize=(12, 12))
disp.plot(
    ax=ax, 
    xticks_rotation=45, 
    colorbar=False
)
plt.title("Finetuned CLIP — Zero-Shot Classification")
ax.tick_params(axis='x', labelsize=8)
ax.tick_params(axis='y', labelsize=8)
plt.tight_layout()
plt.savefig(
    RESULTS_DIR / "confusion_matrix_finetuned.pdf",
    format="pdf",
    bbox_inches="tight"
)
