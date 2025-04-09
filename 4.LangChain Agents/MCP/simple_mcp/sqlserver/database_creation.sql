-- Criar o banco de dados
CREATE DATABASE MercadoFinanceiro;
GO

-- Usar o banco de dados
USE MercadoFinanceiro;
GO

-- Criar a tabela para armazenar os dados históricos
CREATE TABLE HistoricoAcoes (
    ID INT IDENTITY(1,1) PRIMARY KEY,
    Ticker NVARCHAR(10),
    Data DATE,
    Abertura DECIMAL(10,2),
    Alta DECIMAL(10,2),
    Baixa DECIMAL(10,2),
    Fechamento DECIMAL(10,2),
    Volume BIGINT,
    CONSTRAINT UC_HistoricoAcoes_Ticker_Data UNIQUE (Ticker, Data)
);
GO

-- Criar índices para melhor performance
CREATE INDEX IX_HistoricoAcoes_Ticker ON HistoricoAcoes(Ticker);
CREATE INDEX IX_HistoricoAcoes_Data ON HistoricoAcoes(Data);
GO

-- Verificar os últimos 10 registros
SELECT TOP 10 * 
FROM HistoricoAcoes 
ORDER BY Data DESC;
GO

-- Contar total de registros
SELECT COUNT(*) as TotalRegistros 
FROM HistoricoAcoes;
GO

-- Contar registros por ação
SELECT Ticker, COUNT(*) as Registros 
FROM HistoricoAcoes 
GROUP BY Ticker;
GO

-- Média de fechamento por ação
SELECT 
    Ticker,
    AVG(Fechamento) as MediaFechamento,
    MIN(Fechamento) as MenorFechamento,
    MAX(Fechamento) as MaiorFechamento
FROM HistoricoAcoes
GROUP BY Ticker;
GO

-- Variação diária
SELECT 
    Ticker,
    Data,
    Fechamento,
    LAG(Fechamento) OVER (PARTITION BY Ticker ORDER BY Data) as FechamentoAnterior,
    ((Fechamento - LAG(Fechamento) OVER (PARTITION BY Ticker ORDER BY Data)) / 
     LAG(Fechamento) OVER (PARTITION BY Ticker ORDER BY Data)) * 100 as VariacaoDiaria
FROM HistoricoAcoes
ORDER BY Ticker, Data;
GO
