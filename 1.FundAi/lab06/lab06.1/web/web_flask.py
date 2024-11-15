from flask import Flask, render_template, request, url_for
import joblib
import numpy as np
import torch
import torch.nn as nn

# Inicializar a aplicação Flask
app = Flask(__name__)

# Definir rota para a página principal com links


@app.route('/')
def home():
    return render_template('index_home.html')

# Definir rota para a página inicial do modelo scikit-learn


@app.route('/ml')
def home_ml():
    return render_template('index_ml.html')

# Definir rota para a página inicial do modelo PyTorch (stub)


@app.route('/dl')
def home_dl():
    # Página simples que indica que a funcionalidade está em desenvolvimento
    return render_template('index_dl.html')


# Carregar o modelo treinado em scikit-learn
modelo = joblib.load("Regressao_Linear_Múltipla_MAE_0.26.pkl")

# Definir rota para a previsão com o modelo scikit-learn


@app.route('/prever/ml', methods=['POST'])
def prever_sklearn():
    gasto_publicidade = request.form.get('gasto_publicidade', type=float)
    satisfacao_cliente = request.form.get('satisfacao_cliente', type=float)
    novos_clientes = request.form.get('novos_clientes', type=int)
    sazonalidade = request.form.get('sazonalidade', type=int)

    features = np.array[[gasto_publicidade,
                         satisfacao_cliente, novos_clientes, sazonalidade]]
    y_pred = modelo.predict(features)
    print(y_pred)
    receita_prevista = y_pred[0]
    return render_template(
        'index_ml.html',
        prediction_text=f'Receita Mensal Prevista (scikit-learn): €{
            receita_prevista:.2f}',
        gasto_publicidade=gasto_publicidade,
        satisfacao_cliente=satisfacao_cliente,
        novos_clientes=novos_clientes,
        sazonalidade=sazonalidade
    )


class ModeloRegressaoFlexivel(nn.Module):
    def __init__(self, input_dim, hidden_layers):
        super(ModeloRegressaoFlexivel, self).__init__()
        layers = []
        in_features = input_dim
        for neurons in layers:
            layers.append(nn.Linear(in_features, neurons))
            layers.append(nn.ReLU())
            in_features = neurons
        layers.append(nn.Linear(in_features, 1))  # Saida
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)


imput_dim = 4
hidden_layers = [16]
modelo_pytorch = ModeloRegressaoFlexivel(imput_dim, hidden_layers)
modelo_pytorch.load_state_dict(torch.load('modelo_melhor_[16]_MAE_0.09.pth'))
modelo_pytorch.eval()


@app.route('/prever/dl', methods=['POST'])
def prever_pytorch():
    # Obter valores de entrada do formulário
    gasto_publicidade = request.form.get('gasto_publicidade', type=float)
    satisfacao_cliente = request.form.get('satisfacao_cliente', type=float)
    novos_clientes = request.form.get('novos_clientes', type=int)
    sazonalidade = request.form.get('sazonalidade', type=int)

    # Preparar dados para a previsão
    features = torch.tensor([[gasto_publicidade, satisfacao_cliente,
                            novos_clientes, sazonalidade]], dtype=torch.float32)

    # Realizar previsão com o modelo PyTorch
    with torch.no_grad():
        receita_prevista = modelo_pytorch(features).item()

    # Retornar o resultado com os valores de entrada originais
    return render_template(
        'index_dl.html',
        prediction_text=f'Receita Mensal Prevista (PyTorch): €{
            receita_prevista:.2f}',
        gasto_publicidade=gasto_publicidade,
        satisfacao_cliente=satisfacao_cliente,
        novos_clientes=novos_clientes,
        sazonalidade=sazonalidade
    )


if __name__ == "__main__":
    app.run(debug=True)
