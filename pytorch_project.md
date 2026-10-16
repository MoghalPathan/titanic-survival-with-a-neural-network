# PyTorch Mini Project: Titanic Survival with a Neural Network

**Goal:** solve the same Titanic survival problem as your scikit-learn project, but with a neural network you build and train yourself. This lets you directly compare a classic ML model (Random Forest) against a deep learning approach on the exact same data — a genuinely useful comparison to talk about in interviews.

**Setup**
```python
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.metrics import accuracy_score, classification_report

torch.manual_seed(42)
df = pd.read_csv("train_cleaned.csv")
```

---

## 1. Prepare the data (reuse the same preprocessing idea as scikit-learn)
```python
X = df[["Pclass", "Sex", "Age", "SibSp", "Parch", "Fare", "Embarked", "FamilySize"]]
y = df["Survived"].values

num_cols = ["Age", "Fare", "SibSp", "Parch", "FamilySize"]
cat_cols = ["Pclass", "Sex", "Embarked"]
preprocess = ColumnTransformer([
    ("num", StandardScaler(), num_cols),
    ("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols),
])

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

X_train_t = preprocess.fit_transform(X_train).astype("float32")   # fit ONLY on train
X_test_t = preprocess.transform(X_test).astype("float32")          # transform test, don't refit
print(X_train_t.shape, X_test_t.shape)
```
Note: PyTorch needs plain NumPy/tensor arrays, not DataFrames — this is why the ColumnTransformer output (already a NumPy array) is convenient here.

## 2. Convert to tensors and build a DataLoader
```python
X_train_t = torch.tensor(X_train_t)
X_test_t = torch.tensor(X_test_t)
y_train_t = torch.tensor(y_train, dtype=torch.long)
y_test_t = torch.tensor(y_test, dtype=torch.long)

train_loader = DataLoader(TensorDataset(X_train_t, y_train_t), batch_size=16, shuffle=True)
```

## 3. Define the network
```python
class TitanicNet(nn.Module):
    def __init__(self, in_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, 32), nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(32, 16), nn.ReLU(),
            nn.Linear(16, 2),     # 2 output classes: died (0), survived (1)
        )
    def forward(self, x):
        return self.net(x)

model = TitanicNet(X_train_t.shape[1])
print(model)
```
`Dropout(0.2)` randomly zeroes 20% of neurons during training — a regularization trick to reduce overfitting, since this dataset is small.

## 4. Training loop
```python
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

for epoch in range(30):
    model.train()
    total_loss = 0
    for xb, yb in train_loader:
        pred = model(xb)
        loss = criterion(pred, yb)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * len(xb)

    if epoch % 10 == 0:
        print(f"epoch {epoch}: loss {total_loss/len(train_loader.dataset):.4f}")
```

## 5. Evaluate
```python
model.eval()
with torch.no_grad():
    preds = model(X_test_t).argmax(dim=1)

acc = accuracy_score(y_test_t, preds)
print("Test accuracy:", acc)
print(classification_report(y_test_t, preds))
```

## 6. Save and reload
```python
torch.save(model.state_dict(), "titanic_nn.pt")

loaded = TitanicNet(X_train_t.shape[1])
loaded.load_state_dict(torch.load("titanic_nn.pt"))
loaded.eval()

with torch.no_grad():
    same = torch.allclose(model(X_test_t), loaded(X_test_t))
print("Reloaded model gives identical output:", same)
```

---

## Your task
1. Type and run all 6 steps on the real Titanic dataset.
2. Compare this network's test accuracy to your scikit-learn Random Forest's accuracy on the **same train/test split** — which wins? (For a dataset this small and simple, don't be surprised if Random Forest wins — neural nets usually need more data to show an advantage. That's a real, useful insight to understand, not a failure.)
3. Try changing the architecture: add a layer, change hidden sizes (32→64), or remove `Dropout` — does accuracy change?
4. Try a different `lr` (0.01 vs 0.0001) and watch how the loss curve changes.
5. Plot the training loss per epoch with matplotlib (you have the skills from the matplotlib project).

**When done:** tell me your test accuracy and which model (Random Forest vs this network) performed better, and your guess at why.
