import pandas as pd
import numpy as np

# Testando numpy
array = np.array([1, 2, 3, 4, 5])
print("Array numpy:", array)
print("Soma do array numpy:", np.sum(array))

# Testando pandas
data = {
    'Nome': ['Ana', 'Bruno', 'Carlos', 'Diana'],
    'Idade': [23, 35, 45, 28]
}
df = pd.DataFrame(data)
print("\nDataFrame pandas:")
print(df)

print("\nEstatísticas descritivas do DataFrame pandas:")
print(df.describe())