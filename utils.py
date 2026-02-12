
import torch
from torch.utils.data import WeightedRandomSampler
from collections import Counter
import pandas as pd
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from torch.utils.data import Dataset
import seaborn as sns
from sklearn.metrics import confusion_matrix

class AlbWrapper(Dataset):
    def __init__(self, dataset, transform):
        self.ds = dataset
        self.transform = transform
        self.classes = dataset.classes
        self.samples = dataset.samples

    def __getitem__(self, idx):
        path, label = self.ds.samples[idx]
        img = np.array(Image.open(path).convert("RGB"))
        img = self.transform(image=img)["image"]
        return img, label

    def __len__(self):
        return len(self.ds)

def get_weighted_sampler(dataset):
    label_counts = Counter([label for _, label in dataset])
    weights = [1.0 / label_counts[label] for _, label in dataset]
    return WeightedRandomSampler(weights, len(weights))

def save_predictions(loader, model, device, classes, csv_name):
    model.eval()
    records = []
    with torch.no_grad():
        for xb, yb in tqdm(loader, desc=f"[Saving {csv_name}]"):
            xb = xb.to(device)
            preds = model(xb).cpu().argmax(1)
            for i in range(len(preds)):
                path = loader.dataset.samples[i][0]
                records.append((path.split('/')[-1], classes[yb[i]], classes[preds[i]]))
    pd.DataFrame(records, columns=["filename", "true_label", "predicted_label"]).to_csv(csv_name, index=False)

def plot_learning_curve(history):
    plt.figure(figsize=(10, 4))
    plt.subplot(1, 2, 1)
    plt.plot(history['train_loss'], label='Train Loss')
    plt.plot(history['val_loss'], label='Val Loss')
    plt.title("Loss"); plt.grid(); plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(history['train_acc'], label='Train Acc')
    plt.plot(history['val_acc'], label='Val Acc')
    plt.title("Accuracy"); plt.grid(); plt.legend()

    plt.tight_layout()
    plt.savefig("learning_curve.png")
    plt.close()

def plot_confusion_matrix(y_true, y_pred, class_names, filename="confusion_matrix.png"):
    cm = confusion_matrix(y_true, y_pred, labels=range(len(class_names)))
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap="Blues", xticklabels=class_names, yticklabels=class_names)
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title("Confusion Matrix")
    plt.tight_layout()
    plt.savefig(filename)
    plt.close()


from sklearn.metrics import classification_report

def print_classification_report(y_true, y_pred, class_names, title="Classification Report"):
    print(f"\n\n{title}")
    report = classification_report(y_true, y_pred, target_names=class_names)
    print(report)
    with open(title.replace(" ", "_").lower() + ".txt", "w") as f:
        f.write(report)
