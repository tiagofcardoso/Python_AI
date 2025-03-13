# [1] Importações necessárias
from transformers import BertTokenizer, BertForSequenceClassification
from transformers import Trainer, TrainingArguments
from datasets import load_dataset

###############################################
# [2] Carregar Dataset
###############################################
dataset = load_dataset("imdb")
print(dataset)

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
tokenized_datasets = dataset.map(tokenize_function, batched=True)

###############################################
# [6] Carregar Modelo Pré-Treinado
###############################################
model = BertForSequenceClassification.from_pretrained("bert-base-uncased")

###############################################
# [7] Definir Argumentos de Treino
###############################################
training_args = TrainingArguments(
    output_dir="./results",            # Diretório para guardar artefactos (checkpoints, logs)
    evaluation_strategy="epoch",       # Avaliar ao final de cada época
    save_strategy="epoch",             # Guardar modelo ao final de cada época
    learning_rate=2e-5,                # Taxa de aprendizagem recomendada para BERT
    per_device_train_batch_size=16,    # Tamanho do batch de treino
    per_device_eval_batch_size=64,     # Tamanho do batch de avaliação
    num_train_epochs=3,                # Número de épocas de treino
    weight_decay=0.01,                 # Fator de decaimento de pesos (regularização)
)


###############################################
# [8] Instanciar o Trainer
###############################################
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_datasets["train"],
    eval_dataset=tokenized_datasets["test"],
)

###############################################
# [9] Treinar o Modelo
###############################################
trainer.train()

###############################################
# [10] (Opcional) Avaliar o Modelo
###############################################
eval_results = trainer.evaluate()
print("\nAvaliação Final:", eval_results)

###############################################
# [11] (Opcional) Inferência em Novos Exemplos
###############################################
comentario = "This movie was absolutely fantastic!"
inputs = tokenizer(comentario, return_tensors="pt", truncation=True, padding=True)
outputs = model(**inputs)
logits = outputs.logits
predicted_class = logits.argmax(dim=1).item()

print("Comentário:", comentario)
print("Classe prevista:", "Positivo" if predicted_class == 1 else "Negativo")
