from flask import Flask, render_template, request, url_for
import joblib
import numpy as np

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
modelo = joblib.load('Regressão Linear Multipla_MAE_0.26.pkl')

# Definir rota para a previsão com o modelo scikit-learn


@app.route('/prever/ml', methods=['POST'])
def prever_skilearn():
    gasto_publicidade = float(request.form.get('gasto_publicidade'))
    satisfacao_cliente = float(request.form.get('satisfacao_cliente'))
    novos_clientes = int(request.form.get('novos_clientes'))
    sazonalidade = int(request.form.get('sazonalidade'))

    features = np.array(
        [gasto_publicidade, satisfacao_cliente, novos_clientes, sazonalidade]
    )
    y_pred = modelo.predict(features)
    print(y_pred)
    receita_prevista = y_pred[0]
    return render_template('index_ml.html',
                           prediction_text=f'A receita prevista é de ${receita_prevista:.2f}', 
                           gasto_publicidade=float(request.form.get('gasto_publicidade')), 
                           satisfacao_cliente=float( request.form.get('satisfacao_cliente')),
                           novos_clientes=int(request.form.get('novos_clientes')),
                           sazonalidade=int(request.form.get('sazonalidade')))


if __name__ == "__main__":
    app.run(debug=True)
