import json
from typing import Dict, Any
from mcp_server import get_server

class FinancialAnalyser:
    def __init__(self):
        self.server = get_server()
        
    def process_query(self, user_query: str) -> Dict[str, Any]:
        """Processa a consulta do usuário e decide qual ferramenta usar"""
        # Sistema de prompt para o LLM entender a intenção do usuário
        system_prompt = """
        Você é um assistente especializado em análise de dados financeiros. 
        Sua função é interpretar consultas do usuário e determinar se elas requerem acesso ao banco de dados.
        
        Se a consulta for sobre dados históricos de ações, análise de mercado ou qualquer informação 
        que precise de dados do banco, responda com um JSON contendo:
        - database_query: true
        E um dos seguintes campos:
        - sql_query: a consulta SQL apropriada (se for uma consulta direta)
        - analyze_stock: {ticker: "SÍMBOLO", days: NÚMERO_DE_DIAS} (se for análise de ação)
        - best_price: {ticker: "SÍMBOLO"} (se for consulta sobre o melhor preço)
        - worst_price: {ticker: "SÍMBOLO"} (se for consulta sobre o pior preço)
        
        Se a consulta não precisar de dados do banco, responda com um JSON contendo:
        - database_query: false
        - response: sua resposta direta à pergunta
        
        IMPORTANTE: Você está consultando um banco de dados SQL Server. Use a sintaxe correta do SQL Server:
        - Use TOP em vez de LIMIT
        - Use ORDER BY para ordenação
        - Para datas, use o formato YYYY-MM-DD
        - Para consultas de máximo/mínimo, use funções como MAX() e MIN()
        
        Exemplos de tabelas disponíveis:
        - HistoricoAcoes (colunas: ID, Ticker, Data, Abertura, Alta, Baixa, Fechamento, Volume)
        - Empresas (colunas: ID, Nome, Setor, Descricao)
        
        Exemplos de consultas SQL válidas:
        1. Para encontrar o melhor preço (máxima) de uma ação:
           SELECT TOP 1 Ticker, Data, Alta AS MelhorPreco FROM HistoricoAcoes WHERE Ticker = 'CAOS' ORDER BY Alta DESC
        
        2. Para encontrar o pior preço (mínima) de uma ação:
           SELECT TOP 1 Ticker, Data, Baixa AS PiorPreco FROM HistoricoAcoes WHERE Ticker = 'CAOS' ORDER BY Baixa ASC
        
        3. Para listar os últimos 5 registros de uma ação:
           SELECT TOP 5 * FROM HistoricoAcoes WHERE Ticker = 'CAOS' ORDER BY Data DESC
        
        Exemplos de consultas e respostas:
        1. "Qual o melhor preço da ação CAOS?"
           {"database_query": true, "best_price": {"ticker": "CAOS"}}
        
        2. "Quando a ação TITI atingiu seu menor valor?"
           {"database_query": true, "worst_price": {"ticker": "TITI"}}
        
        3. "Analise a ação GOOGL nos últimos 60 dias"
           {"database_query": true, "analyze_stock": {"ticker": "GOOGL", "days": 60}}
        """
        
        # Consultar o LLM para entender a intenção do usuário
        llm_response = self.server.query_llm(user_query, system_prompt)
        
        if llm_response["status"] == "error":
            return {
                "status": "error",
                "error": llm_response["error"]
            }
        
        # Extrair a resposta do LLM
        llm_data = llm_response["data"]
        
        # Se o LLM identificar que é uma consulta de banco de dados
        if "database_query" in llm_data and llm_data["database_query"]:
            # Extrair a consulta SQL ou parâmetros de análise
            if "sql_query" in llm_data:
                # Executar consulta SQL direta
                return self.server.execute_db_query(llm_data["sql_query"])
            elif "analyze_stock" in llm_data:
                # Analisar ação específica
                ticker = llm_data["analyze_stock"].get("ticker")
                days = llm_data["analyze_stock"].get("days", 30)
                if ticker:
                    return self.server.analyze_stock(ticker, days)
            elif "best_price" in llm_data:
                # Obter o melhor preço de uma ação
                ticker = llm_data["best_price"].get("ticker")
                if ticker:
                    return self.server.get_best_price(ticker)
            elif "worst_price" in llm_data:
                # Obter o pior preço de uma ação
                ticker = llm_data["worst_price"].get("ticker")
                if ticker:
                    return self.server.get_worst_price(ticker)
        
        # Se não for uma consulta de banco de dados ou não tiver informações suficientes
        return {
            "status": "success",
            "source": "llm",
            "response": llm_data.get("response", "Não entendi sua consulta. Pode reformular?")
        }

    def format_result(self, result: Dict[str, Any]) -> str:
        """Formata o resultado para exibição ao usuário"""
        if result["status"] == "error":
            return f"Erro: {result.get('error', 'Ocorreu um erro desconhecido')}"
        
        if "source" in result and result["source"] == "llm":
            return f"Resposta: {result['response']}"
        
        if "recommendation" in result:
            # Formatação para análise de ação
            output = []
            output.append(f"Análise da ação {result['ticker']} (período: {result['period']})")
            output.append(f"Preço médio: {result['metrics']['avg_price']:.2f}")
            output.append(f"Preço máximo: {result['metrics']['max_price']:.2f}")
            if 'max_price_date' in result['metrics']:
                output.append(f"Data do preço máximo: {result['metrics']['max_price_date']}")
            output.append(f"Preço mínimo: {result['metrics']['min_price']:.2f}")
            if 'min_price_date' in result['metrics']:
                output.append(f"Data do preço mínimo: {result['metrics']['min_price_date']}")
            output.append(f"Desvio padrão: {result['metrics']['std_dev']:.2f}")
            output.append(f"Volume médio: {result['metrics']['volume_avg']:.2f}")
            output.append(f"Recomendação: {result['recommendation']}")
            output.append(f"Baseado em {result['data_points']} pontos de dados")
            return "\n".join(output)
        
        # Formatação para consulta genérica
        output = []
        output.append(f"Resultado: {result.get('message', '')}")
        
        if "data" in result and result["data"]:
            output.append(f"Encontrados {result.get('count', len(result['data']))} registros:")
            
            # Mostrar apenas os primeiros 5 resultados para não sobrecarregar o terminal
            for i, row in enumerate(result["data"][:5]):
                output.append(f"\nRegistro {i+1}:")
                for key, value in row.items():
                    output.append(f"  {key}: {value}")
            
            if len(result["data"]) > 5:
                output.append(f"\n... e mais {len(result['data']) - 5} registros")
        
        return "\n".join(output)

# Interface de linha de comando
def main():
    print("=== Analisador Financeiro ===")
    print("Digite 'sair' para encerrar o programa\n")
    
    analyser = FinancialAnalyser()
    
    while True:
        user_input = input("\nSua consulta: ")
        if user_input.lower() in ["sair", "exit", "quit"]:
            print("Encerrando o programa...")
            break
        
        print("\nProcessando...")
        result = analyser.process_query(user_input)
        formatted_result = analyser.format_result(result)
        print(f"\n{formatted_result}")

if __name__ == "__main__":
    main()