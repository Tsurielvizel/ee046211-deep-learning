
import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
import torchvision
import matplotlib.pyplot as plt
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
import torch.nn.init as init
import math
# %matplotlib ipympl
# %matplotlib inline

seed = 211
np.random.seed(seed)
torch.manual_seed(seed)

#device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#print(f"Using device: {device}")
#if torch.cuda.is_available():
 #   torch.cuda.manual_seed(seed)
  #  torch.backends.cudnn.deterministic = True
   # torch.backends.cudnn.benchmark = False

# loading the data
col_names = ['fLength', 'fWidth', 'fSize', 'fConc', 'fConc1', 'fAsym',  'fM3Long', 'fM3Trans', 'fAlpha', 'fDist', 'class']
feature_names = ['fLength', 'fWidth', 'fSize', 'fConc', 'fConc1', 'fAsym',  'fM3Long', 'fM3Trans', 'fAlpha', 'fDist']
#data = pd.read_csv("./magic04.data", names=col_names)
data = pd.read_csv("/content/magic04.data", names=col_names)
X = data[feature_names]
Y = data['class']
data.head()

# separate to train, test
x_train, x_test, y_train, y_test = train_test_split(X,Y,test_size=0.2,random_state=36)
x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size=0.125,random_state=18)

#3 pre-processing and converting labels to integers

y_train_np = np.array([0 if y == 'g' else 1 for y in y_train]).astype(int)
y_val_np = np.array([0 if y == 'g' else 1 for y in y_val]).astype(int)
y_test_np = np.array([0 if y == 'g' else 1 for y in y_test]).astype(int)

#4 training a Logistic Regression baseline - complete the code with your variables
logstic_model = LogisticRegression(solver='lbfgs', max_iter=1000) #
X_train_val_lr = pd.concat([x_train, x_val])
y_train_val_lr = np.concatenate([y_train_np, y_val_np])

y_pred = logstic_model.fit(X_train_val_lr, y_train_val_lr).predict(X_train_val_lr)
print("Logistic Regression Model accuracy =" , logstic_model.score(x_test, y_test_np))

#5
#  create TensorDataset from numpy arrays

# make the numpy Tensors

x_train_tensor = torch.tensor(x_train.values, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train_np, dtype=torch.long)

x_val_tensor = torch.tensor(x_val.values, dtype=torch.float32)
y_val_tensor = torch.tensor(y_val_np, dtype=torch.long)

x_test_tensor = torch.tensor(x_test.values, dtype=torch.float32)
y_test_tensor = torch.tensor(y_test_np, dtype=torch.long)

# making TensorDataset
train_dataset = TensorDataset(x_train_tensor, y_train_tensor)
val_dataset = TensorDataset(x_val_tensor, y_val_tensor)
test_dataset = TensorDataset(x_test_tensor, y_test_tensor)

batch_size = 64

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size)
test_loader = DataLoader(test_dataset, batch_size=batch_size)

#6
# Define the MLP  model

class MLPClassifier(nn.Module):
    def __init__(self, input_size=10, dropout_rate=0.2):
        super(MLPClassifier, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(input_size, 256),
            nn.BatchNorm1d(256),
            nn.LeakyReLU(0.01),
            nn.Dropout(dropout_rate),

            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.LeakyReLU(0.01),
            nn.Dropout(dropout_rate),

            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.LeakyReLU(0.01),
            nn.Dropout(dropout_rate),

            nn.Linear(64, 1)
        )

    def forward(self, x):
        return self.model(x)

# Function to train the model
def train_model(model, train_loader, val_loader, epochs=50, lr=0.001, weight_decay=0.0):
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    criterion = nn.BCEWithLogitsLoss()

    train_losses, val_losses, val_accuracies = [], [], []
    best_val_accuracy_for_this_run = 0.0  # Tracks best accuracy for THIS specific training run
    best_model_state_dict_local = None    # Will store the weights of the model at its best validation accuracy

    for epoch in range(epochs):
        model.train()
        epoch_loss = 0
        for xb, yb in train_loader:
            xb = xb.to(device)
            yb = yb.to(device)
            optimizer.zero_grad()
            out = model(xb).squeeze(1)
            loss = criterion(out, yb.float())
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()

        train_losses.append(epoch_loss / len(train_loader))

        if val_loader:
            model.eval()
            val_loss, correct, total = 0, 0, 0
            with torch.no_grad():
                for xb, yb in val_loader:
                    xb = xb.to(device)
                    yb = yb.to(device)
                    out = model(xb).squeeze(1)
                    loss = criterion(out, yb.float())
                    val_loss += loss.item()

                    preds = (torch.sigmoid(out) > 0.5).long()
                    correct += (preds == yb).sum().item()
                    total += yb.size(0)

            current_val_accuracy = correct / total
            val_losses.append(val_loss / len(val_loader))
            val_accuracies.append(current_val_accuracy)

            if current_val_accuracy > best_val_accuracy_for_this_run:
                best_val_accuracy_for_this_run = current_val_accuracy
                best_model_state_dict_local = model.state_dict()

            print(f"Epoch {epoch+1}/{epochs} - Train Loss: {train_losses[-1]:.4f} | "
                  f"Val Loss: {val_losses[-1]:.4f} | Val Acc: {val_accuracies[-1]*100:.2f}%")
        else:
            print(f"Epoch {epoch+1}/{epochs} - Train Loss: {train_losses[-1]:.4f}")

    if best_model_state_dict_local:
        model.load_state_dict(best_model_state_dict_local)
        print("Loaded best model weights for this run (from epoch with best validation accuracy).")
    else:
        print("Warning: No improvement in validation accuracy from initial state. Returning model from last epoch.")

    print(f"Best Validation Accuracy reported by train_model: {best_val_accuracy_for_this_run*100:.2f}%")
    return train_losses, val_losses, val_accuracies, best_val_accuracy_for_this_run

model = MLPClassifier()
train_losses, val_losses, val_accuracies, _ = train_model(model, train_loader, val_loader, epochs=100, lr=0.001,weight_decay=0.001)

plt.plot(train_losses, label='Train Loss')
plt.plot(val_losses, label='Validation Loss')
plt.plot(val_accuracies, label='Validation Accuracy')
plt.xlabel('Epoch')
plt.legend()
plt.title("Loss and Accuracy per Epoch")
plt.show()

# Function to evaluate the model
def evaluate(model, test_loader):
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for xb, yb in test_loader:
            xb = xb.to(device)
            yb = yb.to(device)
            out = model(xb).squeeze(1)
            preds = (torch.sigmoid(out) > 0.5).long()
            correct += (preds == yb).sum().item()
            total += yb.size(0)
    acc = correct / total
    print(f"Test Accuracy: {acc*100:.2f}%")
    return acc

# model, hyoer-paramerters and training
# Hyperparameter Optimization
print("\n--- Starting Hyperparameter Optimization ---")

best_overall_accuracy = 0.0
best_hyperparams = {}
best_model_state_dict = None

learning_rates = [0.0007, 0.0008]

epochs_options = [150]

dropout_rates = [0.15, 0.3]

#weight_decay_options = [0.0, 0.0001, 0.0005, 0.001]
weight_decay_options = [0.0]

# Add a new nested loop for weight_decay
for lr in learning_rates:
    for epochs in epochs_options:
        for dropout in dropout_rates:
            for wd in weight_decay_options: # <-- NEW LOOP HERE
                print(f"\nTesting with LR: {lr}, Epochs: {epochs}, Dropout: {dropout}, Weight Decay: {wd}")
                temp_model = MLPClassifier(dropout_rate=dropout).to(device)

                _, _, _, current_best_val_accuracy = train_model(temp_model, train_loader, val_loader,
                                                                  epochs=epochs, lr=lr, weight_decay=wd)

                if current_best_val_accuracy > best_overall_accuracy:
                    best_overall_accuracy = current_best_val_accuracy
                    best_hyperparams = {'lr': lr, 'epochs': epochs, 'dropout': dropout, 'weight_decay': wd}
                    best_model_state_dict = temp_model.state_dict()

print("\n--- Hyperparameter Optimization Complete ---")
print(f"Best Hyperparameters found: {best_hyperparams}")
print(f"Best Validation Accuracy during search: {best_overall_accuracy*100:.2f}%")

# Combine train and validation data for final model training
full_train_tensor = torch.tensor(np.vstack([x_train.values, x_val.values]), dtype=torch.float32)
full_label_tensor = torch.tensor(np.concatenate([y_train_np, y_val_np]), dtype=torch.long)

full_dataset = TensorDataset(full_train_tensor, full_label_tensor)
full_loader = DataLoader(full_dataset, batch_size=batch_size, shuffle=True)

# Train the final model on combined Train + Validation set with best hyperparameters
print("\n--- Training final model on combined Train + Validation set with best hyperparameters ---")
final_model = MLPClassifier(dropout_rate=best_hyperparams['dropout']).to(device)
train_losses_final, _, _, _ = train_model(final_model, full_loader, val_loader=None, epochs=best_hyperparams['epochs'], lr=best_hyperparams['lr'],weight_decay=best_hyperparams['weight_decay'])
final_test_accuracy = evaluate(final_model, test_loader)

# example of weight initialization
import torch.nn as nn
class MyModel(nn.Module):
    def __init__(self, parmaeters):
        super(MyModel, self).__init__()
        # model definitions/blocks
        # ...
        # custom initialization
        self.init_weights()

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                # pick initialzation: https://pytorch.org/docs/stable/nn.init.html
                # examples
                # nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                # nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='leaky_relu', a=math.sqrt(5))
                # nn.init.normal_(m.weight, 0, 0.005)
                # don't forget the bias term (m.bias)

    def forward(self, x):
        # ops on x
        # ...
        # output = f(x)
        return output

"""---
### ארכיטקטורה והיפר-פרמטרים של רשת נוירונים (MLP)

המודל הוא רשת עצבית רב-שכבתית (MLP) שתוכננה לאור נתוני ה-MAGIC ועל בסיס אופטימיזציית היפר-פרמטרים.

#### ארכיטקטורה (MLPClassifier):

* **שכבת קלט (Input Layer):** מקבלת 10 פיצ'רים.
* **שכבה נסתרת ראשונה:**
    * **`nn.Linear(10, 256)`**: שכבה לינארית עם 256 נוירונים.
    * **`nn.BatchNorm1d(256)`**: שכבת Batch Normalization לייצוב האימון.
    * **`nn.LeakyReLU(0.01)`**: פונקציית אקטיבציה למניעת Dying ReLUs.
    * **`nn.Dropout(0.15)`**: שכבת Dropout לרגולריזציה ומניעת Overfitting.
* **שכבה נסתרת שנייה:**
    * **`nn.Linear(256, 128)`**: שכבה לינארית עם 128 נוירונים.
    * **`nn.BatchNorm1d(128)`**: שכבת Batch Normalization.
    * **`nn.LeakyReLU(0.01)`**: פונקציית אקטיבציה.
    * **`nn.Dropout(0.15)`**: שכבת Dropout.
* **שכבה נסתרת שלישית:**
    * **`nn.Linear(128, 64)`**: שכבה לינארית עם 64 נוירונים.
    * **`nn.BatchNorm1d(64)`**: שכבת Batch Normalization.
    * **`nn.LeakyReLU(0.01)`**: פונקציית אקטיבציה.
    * **`nn.Dropout(0.15)`**: שכבת Dropout.
* **שכבת פלט (Output Layer):**
    * **`nn.Linear(64, 1)`**: יחידת פלט אחת לסיווג בינארי.

#### היפר-פרמטרים:

היפר-פרמטרים אלו נבחרו באמצעות תהליך אופטימיזציית היפר-פרמטרים על בסיס ביצועי סט הוולידציה:

* **קצב למידה (Learning Rate):** **0.0007**.
* **אופטימייזר:** **Adam** – נבחר בזכות יכולותיו להתמודד עם קצבי למידה משתנים דינמית ולספק ביצועים טובים.
* **פונקציית הפסד (Loss Criterion):** **`nn.BCEWithLogitsLoss()`** – פונקציית הפסד המתאימה לסיווג בינארי, המשלבת את פונקציית הסיגמואיד ומסייעת ליציבות מספרית.
* **מספר אפוכות (Epochs):** **150**.
* **קצב Dropout:** **0.15**.
* **גודל אצווה (Batch Size):** **64** – ערך סטנדרטי המספק איזון בין יציבות אימון למהירות.
* **Weight Decay:** **0.0** (לא נעשה שימוש ב-L2 regularization).

#### שיקולי עיצוב לשיפור ביצועים:

* **מעבר מארכיטקטורה עמוקה לרחבה:** בתהליך האופטימיזציה, נמצא כי רשת רחבה (עם יותר נוירונים בשכבות ופחות שכבות עמוקות) הניבה תוצאות טובות יותר עבור מספר הפיצ'רים המוגבל של הנתונים. זה מאפשר ללכוד מגוון רחב יותר של יחסים בתוך כל שכבה.
* **Batch Normalization:** שכבות אלו הוכנסו בכל שכבה נסתרת לייצוב תהליך האימון, איפשרו שימוש ב-learning rates גבוהים יותר וקידמו התכנסות מהירה יותר.
* **Leaky ReLU:** נבחרה כפונקציית אקטיבציה כדי למנוע את בעיית ה-"Dying ReLUs" ולאפשר זרימת גרדיאנטים יעילה יותר במהלך ה-backpropagation.
* **Dropout:** שימוש בשכבות Dropout (עם קצב של 0.15) היה קריטי למניעת Overfitting ולאימון מודל שמכליל היטב לנתונים חדשים. זה מאלץ את הרשת ללמוד ייצוגים יציבים וחזקים יותר.
* **אופטימייזר Adam:** בחירה באופטימייזר זה תרמה להתכנסות מהירה ויעילה, בזכות יכולתו להתאים באופן דינמי את קצב הלמידה לכל פרמטר.

---
"""

# נבחר שתי שיטות אתחול נוספות לבדיקה
initialization_methods_to_test = {
    'xavier_uniform': 'Xavier Uniform Initialization',
    'normal_0_01': 'Normal Initialization (mean=0, std=0.01)',
    # 'kaiming_normal_leaky_relu': 'Kaiming Normal for Leaky ReLU', # זה לרוב אתחול ברירת המחדל של פייטרוץ'
    'uniform_neg_pos_0_1': 'Uniform Initialization (-0.1 to 0.1)'
}

results = {
    'Baseline (PyTorch Default Init)': baseline_test_accuracy * 100
}

# נריץ ניסויים עבור כל שיטת אתחול
for method_key, method_name in initialization_methods_to_test.items():
    print(f"\n--- Training with {method_name} ---")

    # חשוב: לאפס את ה-seeds כדי לוודא שחזוריות עבור כל הרצת אתחול
    # זה מבטיח שכל אתחול מתחיל מאותו מצב התחלתי של רנדומיות
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)

    # 1. יצירת מודל חדש והעברתו ל-GPU
    current_model = MLPClassifier(dropout_rate=best_hyperparams['dropout']).to(device)

    # 2. קריאה לפונקציית init_weights של המודל כדי לאתחל אותו בשיטה הספציפית
    current_model.init_weights(method=method_key)

    # 3. אימון המודל עם ההיפר-פרמטרים האופטימליים שנמצאו קודם
    train_losses_current, _, _, _ = train_model(current_model, full_loader, val_loader=None,
                                                epochs=best_hyperparams['epochs'], lr=best_hyperparams['lr'],
                                                weight_decay=best_hyperparams['weight_decay'])

    # 4. הערכה על סט הבדיקה
    current_test_accuracy = evaluate(current_model, test_loader)
    results[method_name] = current_test_accuracy * 100

    # 5. שרטוט עקומת ה-loss עבור האתחול הנוכחי
    plt.figure(figsize=(10, 5))
    plt.plot(train_losses_current, label=f'{method_name} Train Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.title(f"{method_name} Loss Curve (Trained on Train + Validation)")
    plt.grid(True)
    plt.show()

# הדפסת סיכום השוואת תוצאות סופיות
print("\n" + "="*50)
print("              SUMMARY OF TEST ACCURACIES")
print("="*50 + "\n")

for name, acc in results.items():
    print(f"{name}: {acc:.2f}%")

# חישוב והדפסת שינוי מהבייסליין
if 'Baseline (PyTorch Default Init)' in results:
    baseline_acc = results['Baseline (PyTorch Default Init)']
    for name, acc in results.items():
        if name != 'Baseline (PyTorch Default Init)':
            change = acc - baseline_acc
            print(f"Change for {name} vs. Baseline: {change:.2f}%")
