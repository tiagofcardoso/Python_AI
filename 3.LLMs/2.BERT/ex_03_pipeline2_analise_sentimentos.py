from transformers import pipeline

model = pipeline(task ="text-classification", 
                 model="nlptown/bert-base-multilingual-uncased-sentiment")


result = model("a fé na vitória tem que ser inabalável!")

print(result)