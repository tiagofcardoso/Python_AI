# [1] Importações necessárias
import numpy as np
from transformers import BertTokenizer, BertForSequenceClassification, Trainer, TrainingArguments
from dataset import load_dataset

###############################################
# [2] Carregar Dataset
###############################################

dataset = load_dataset("imdb")
print(dataset["train"])
print(dataset["test"])

###############################################
# [3] Criar/Carregar o Tokenizer
###############################################

tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

###############################################
# [4] Definir Função de Tokenização
###############################################


def tokenize_function(examples):
    return tokenizer(examples["text"], padding="max_length", truncation=True)

###############################################
# [5] Aplicar Tokenização ao Dataset
###############################################

tokenized_dataset = dataset.map(tokenize_function, batched=True)

###############################################
# [6] Carregar Modelo Pré-Treinado
###############################################

model = BertForSequenceClassification.from_pretrained("bert-base-uncased")

###############################################
# [7] Definir Argumentos de Treino
###############################################

training_args = TrainingArguments(
    output_dir="./results",
    evaluation_strategy="epoch",
    save_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=64,
    num_train_epochs=3,
    weight_decay=0.01,
)


###############################################
# [8] Instanciar o Trainer
###############################################

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset["train"],
    eval_dataset=tokenized_dataset["test"],
)

###############################################
# [9] Treinar o Modelo
###############################################

trainer.train()

###############################################
# [10] (Opcional) Avaliar o Modelo
###############################################

accuracy_metric = load_metric("accuracy")
def compute_metrics(p):
    preds = np.argmax(p.predictions, axis=1)
    return accuracy_metric.compute(predictions=preds, references=p.label_ids)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset["train"],
    eval_dataset=tokenized_dataset["test"],
    compute_metrics=compute_metrics
)
trainer.train()
eval_results = trainer.evaluate()
print(eval_results)

###############################################
# [11] (Opcional) Inferência em Novos Exemplos
###############################################
# Exemplo simples de inferência
text = "This movie was absolutely fantastic!"
inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True)
outputs = model(**inputs)
logits = outputs.logits
predicted_class = logits.argmax(dim=1).item()
print("Comentário:", text)
print("Classe Prevista:", predicted_class)

