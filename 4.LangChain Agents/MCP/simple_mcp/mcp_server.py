import os
import json
from typing import Dict, Any, List, Optional
import pyodbc
import pandas as pd
from datetime import datetime
from openai import OpenAI

# Configuração da API OpenAI
api_key = os.environ.get("OPENAI_API_KEY", "")
client = OpenAI(api_key=api_key)

# Configuração da conexão com SQL Server


class SQLServerConnection:
    def __init__(self):
        self.connection_string = (
            "DRIVER={ODBC Driver 17 for SQL Server};"
            "SERVER=localhost,1433;"
            "DATABASE=MercadoFinanceiro;"
            "UID=SA;"
            f"PWD={os.environ.get('SQL_PASSWORD', '')};"
        )

    def connect(self):
        return pyodbc.connect(self.connection_string)

# Agente para consulta ao banco de dados


class DatabaseAgent:
    def __init__(self):
        self.db = SQLServerConnection()

    def query_database(self, query: str) -> Dict[str, Any]:
        """Executa uma consulta SQL e retorna os resultados"""
        try:
            with self.db.connect() as conn:
                cursor = conn.cursor()
                cursor.execute(query)

                # Se não houver resultados (como em INSERT, UPDATE, DELETE)
                if not cursor.description:
                    return {
                        "status": "success",
                        "message": f"Operação concluída. Linhas afetadas: {cursor.rowcount}",
                        "data": []
                    }

                columns = [column[0] for column in cursor.description]
                results = []
                for row in cursor.fetchall():
                    results.append(dict(zip(columns, row)))

                return {
                    "status": "success",
                    "count": len(results),
                    "data": results
                }
        except Exception as e:
            return {
                "status": "error",
                "error": str(e)
            }

    def analyze_stock(self, ticker: str, days: int = 30) -> Dict[str, Any]:
        """Analisa dados históricos de uma ação"""
        query = f"""
        SELECT *
        FROM HistoricoAcoes
        WHERE Ticker = '{ticker}'
        AND Data >= DATEADD(day, -{days}, GETDATE())
        ORDER BY Data
        """

        result = self.query_database(query)
        if result["status"] == "error" or not result.get("data"):
            return {
                "status": "error",
                "message": f"Não foi possível encontrar dados para a ação {ticker}"
            }

        # Converter para DataFrame para análise
        df = pd.DataFrame(result["data"])

        # Converter colunas numéricas para float para evitar problemas com decimal.Decimal
        if "Fechamento" in df.columns:
            df["Fechamento"] = df["Fechamento"].astype(float)
        if "Alta" in df.columns:
            df["Alta"] = df["Alta"].astype(float)
        if "Baixa" in df.columns:
            df["Baixa"] = df["Baixa"].astype(float)
        if "Volume" in df.columns:
            df["Volume"] = df["Volume"].astype(float)

        # Calcular métricas
        metrics = {
            "avg_price": float(df["Fechamento"].mean()) if "Fechamento" in df.columns else 0,
            "std_dev": float(df["Fechamento"].std()) if "Fechamento" in df.columns else 0,
            "max_price": float(df["Alta"].max()) if "Alta" in df.columns else 0,
            "min_price": float(df["Baixa"].min()) if "Baixa" in df.columns else 0,
            "volume_avg": float(df["Volume"].mean()) if "Volume" in df.columns else 0
        }

        # Encontrar a data do preço máximo e mínimo
        if "Alta" in df.columns and len(df) > 0:
            max_idx = df["Alta"].idxmax()
            max_date = df.loc[max_idx,
                              "Data"] if "Data" in df.columns else None
            metrics["max_price_date"] = max_date

        if "Baixa" in df.columns and len(df) > 0:
            min_idx = df["Baixa"].idxmin()
            min_date = df.loc[min_idx,
                              "Data"] if "Data" in df.columns else None
            metrics["min_price_date"] = min_date

        # Gerar recomendação
        volatility = metrics["std_dev"] / \
            metrics["avg_price"] if metrics["avg_price"] > 0 else 0
        if volatility > 0.02:
            recommendation = "Alta volatilidade - Cautela recomendada"
        elif volatility > 0.01:
            recommendation = "Volatilidade moderada - Monitorar"
        else:
            recommendation = "Baixa volatilidade - Estável"

        return {
            "status": "success",
            "ticker": ticker,
            "period": f"{days} dias",
            "metrics": metrics,
            "recommendation": recommendation,
            "data_points": len(df)
        }

# Servidor MCP para interagir com o LLM e banco de dados


class MCPServer:
    def __init__(self):
        self.db_agent = DatabaseAgent()
        self.model = "gpt-4o"

    def query_llm(self, prompt: str, system_prompt: str = None) -> Dict[str, Any]:
        """Consulta o modelo de linguagem da OpenAI"""
        try:
            messages = []

            if system_prompt:
                messages.append({"role": "system", "content": system_prompt})

            messages.append({"role": "user", "content": prompt})

            response = client.chat.completions.create(
                model=self.model,
                messages=messages,
                response_format={"type": "json_object"}
            )

            # Extrair e analisar a resposta JSON
            content = response.choices[0].message.content
            return {
                "status": "success",
                "data": json.loads(content)
            }

        except Exception as e:
            return {
                "status": "error",
                "error": str(e)
            }

    def get_best_price(self, ticker: str) -> Dict[str, Any]:
        """Obtém o melhor preço (máxima) de uma ação"""
        query = f"""
        SELECT TOP 1 Ticker, Data, Alta AS MelhorPreco 
        FROM HistoricoAcoes 
        WHERE Ticker = '{ticker}' 
        ORDER BY Alta DESC
        """
        return self.query_database(query)

    def get_worst_price(self, ticker: str) -> Dict[str, Any]:
        """Obtém o pior preço (mínima) de uma ação"""
        query = f"""
        SELECT TOP 1 Ticker, Data, Baixa AS PiorPreco 
        FROM HistoricoAcoes 
        WHERE Ticker = '{ticker}' 
        ORDER BY Baixa ASC
        """
        return self.query_database(query)

    def execute_db_query(self, query: str) -> Dict[str, Any]:
        """Executa uma consulta no banco de dados"""
        return self.db_agent.query_database(query)

    def analyze_stock(self, ticker: str, days: int = 30) -> Dict[str, Any]:
        """Analisa dados de uma ação"""
        return self.db_agent.analyze_stock(ticker, days)

    def get_best_price(self, ticker: str) -> Dict[str, Any]:
        """Obtém o melhor preço de uma ação"""
        return self.db_agent.get_best_price(ticker)

    def get_worst_price(self, ticker: str) -> Dict[str, Any]:
        """Obtém o pior preço de uma ação"""
        return self.db_agent.get_worst_price(ticker)

# Função para criar e retornar uma instância do servidor


def get_server():
    return MCPServer()


if __name__ == "__main__":
    print("Este módulo fornece serviços de conexão com OpenAI e banco de dados.")
    print("Importe-o em outro módulo para utilizar suas funcionalidades.")
