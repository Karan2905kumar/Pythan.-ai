
import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer
from transformers import BertTokenizer, BertForSequenceClassification
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from torch.utils.data import DataLoader, Dataset
import torch.nn.functional as F
import pandas as pd

# --------------------------
# FAKE NEWS GENERATOR (GPT-2)
# --------------------------
print("Loading GPT-2 for Fake News Generation...")
gpt2_tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
gpt2_model = GPT2LMHeadModel.from_pretrained("gpt2")

def generate_fake_news(prompt="Breaking:"):
    inputs = gpt2_tokenizer(prompt, return_tensors="pt")
    outputs = gpt2_model.generate(**inputs, max_length=50, do_sample=True)
    text = gpt2_tokenizer.decode(outputs[0], skip_special_tokens=True)
    return text

# Example:
print("\n📰 Generated Fake News Example:")
print(generate_fake_news("Breaking:"))

# --------------------------
# FAKE NEWS DETECTOR (BERT)
# --------------------------

print("\nLoading Dataset and BERT for Detection...")
# Load dataset (you can replace this with any dataset)
df = pd.read_csv("https://raw.githubusercontent.com/datasciencedojo/datasets/master/fake_or_real_news.csv")
df['label'] = df['label'].map({'REAL': 0, 'FAKE': 1})

train_texts, val_texts, train_labels, val_labels = train_test_split(df['text'], df['label'], test_size=0.2)

bert_tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
train_encodings = bert_tokenizer(list(train_texts), truncation=True, padding=True, max_length=512)
val_encodings = bert_tokenizer(list(val_texts), truncation=True, padding=True, max_length=512)

class NewsDataset(Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels
    def __getitem__(self, idx):
        return {
            'input_ids': torch.tensor(self.encodings['input_ids'][idx]),
            'attention_mask': torch.tensor(self.encodings['attention_mask'][idx]),
            'labels': torch.tensor(self.labels[idx])
        }
    def __len__(self):
        return len(self.labels)

train_dataset = NewsDataset(train_encodings, list(train_labels))
val_dataset = NewsDataset(val_encodings, list(val_labels))

bert_model = BertForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=2)

train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=8)

optimizer = torch.optim.AdamW(bert_model.parameters(), lr=5e-5)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
bert_model.to(device)

print("\nTraining BERT...")
bert_model.train()
for epoch in range(1):  # For demo, 1 epoch
    for batch in train_loader:
        batch = {k: v.to(device) for k, v in batch.items()}
        outputs = bert_model(**batch)
        loss = outputs.loss
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

print("Evaluating...")
bert_model.eval()
preds, truths = [], []
with torch.no_grad():
    for batch in val_loader:
        batch = {k: v.to(device) for k, v in batch.items()}
        outputs = bert_model(**batch)
        logits = outputs.logits
        preds.extend(torch.argmax(logits, dim=1).cpu().numpy())
        truths.extend(batch['labels'].cpu().numpy())

accuracy = accuracy_score(truths, preds)
print(f"\n✅ Detection Accuracy: {accuracy:.2f}")

# --------------------------
# Sample Detection
# --------------------------
def detect_news(text):
    enc = bert_tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=512)
    enc = {k: v.to(device) for k, v in enc.items()}
    output = bert_model(**enc)
    pred = torch.argmax(output.logits, dim=1).item()
    return "FAKE" if pred == 1 else "REAL"

# Test the detector
sample = generate_fake_news("Just in:")
print(f"\n🧪 Detecting Sample: {sample}")
print(f"Prediction: {detect_news(sample)}")
