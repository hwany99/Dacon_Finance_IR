from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
import pandas as pd
import torch
from torch.optim import AdamW
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
from peft import get_peft_model, LoraConfig

# model & tokenizer
bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16
)

lora_config = LoraConfig(
    r=8,
    target_modules=["q_proj", "o_proj", "k_proj", "v_proj", "gate_proj", "up_proj", "down_proj"],
    task_type="CAUSAL_LM",
)

model_id = "rtzr/ko-gemma-2-9b-it"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = AutoModelForCausalLM.from_pretrained(model_id, quantization_config=bnb_config, device_map='auto')
model = get_peft_model(model, lora_config)
tokenizer = AutoTokenizer.from_pretrained(model_id)
optimizer = AdamW(model.parameters(), lr=5e-5)


# load data
base_dir = './data/'
df = pd.read_csv(base_dir + 'train.csv')

# dataset
class QADataset(Dataset):
    def __init__(self, df, tokenizer, max_length=512):
        self.df = df
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        question = self.df.loc[idx, 'Question']
        answer = self.df.loc[idx, 'Answer']

        inputs = self.tokenizer(
            f"""질문:
{question}

조건:
주어와 서술어를 갖춘 완벽한 문장으로 답변해 주세요.
단, 예산 질문은 금액의 단위를 포함하여 단답형으로 답변해 주세요.

답변:
""", 
            return_tensors="pt", 
            max_length=self.max_length, 
            truncation=True, 
            padding="max_length"
        )
        input_ids = inputs["input_ids"].squeeze(0)
        attention_mask = inputs["attention_mask"].squeeze(0)

        labels = self.tokenizer(
            answer, 
            return_tensors="pt", 
            max_length=self.max_length, 
            truncation=True, 
            padding="max_length"
        )["input_ids"].squeeze(0)
        
        return input_ids, attention_mask, labels
    
train_df, valid_df = df[140:], df[:140]
train_df, valid_df = train_df.reset_index(), valid_df.reset_index()

train_dataset = QADataset(train_df, tokenizer)
train_dataloader = DataLoader(train_dataset, batch_size=4, shuffle=True)

valid_dataset = QADataset(valid_df, tokenizer)
valid_dataloader = DataLoader(valid_dataset, batch_size=4, shuffle=False)


# train
model.train()

epochs = 10
accumulate_grad_batches = 4
min_val_loss = float('inf')

for epoch in range(epochs):
    optimizer.zero_grad()
    running_loss = 0.0
    model.train()
    for step, batch in tqdm(enumerate(train_dataloader), total=len(train_dataloader), desc=f'Epoch {epoch + 1}'):
        input_ids, attention_mask, labels = [b.to(device) for b in batch]
        
        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels
        )
        loss = outputs.loss
        
        loss = loss / accumulate_grad_batches
        loss.backward()

        running_loss += loss.item()
        
        if (step + 1) % accumulate_grad_batches == 0:
            optimizer.step()
            optimizer.zero_grad()

    if len(train_dataloader) % accumulate_grad_batches != 0:
        optimizer.step()
        optimizer.zero_grad()

    avg_train_loss = running_loss / len(train_dataloader)
    print(f"Epoch {epoch+1}, Average Training Loss: {avg_train_loss}")
    
    # Validation
    model.eval()
    running_val_loss = 0.0
    with torch.no_grad():
        for batch in valid_dataloader:
            input_ids, attention_mask, labels = [b.to(device) for b in batch]
            
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels
            )
            val_loss = outputs.loss
            running_val_loss += val_loss.item()

    avg_val_loss = running_val_loss / len(valid_dataloader)
    print(f"Epoch {epoch+1}, Validation Loss: {avg_val_loss}")

    if avg_val_loss < min_val_loss:
        min_val_loss = avg_val_loss

        # Save
        save_dir = "model"
        model.save_pretrained(save_dir)
        tokenizer.save_pretrained(save_dir)
        print("Model saved successfully.")