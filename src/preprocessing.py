import os
import pandas as pd
import numpy as np

def garantir_pastas():
    """Garante que a pasta data/processed exista antes de salvar os dados."""
    os.makedirs('data/processed', exist_ok=True)

def carregar_e_limpar_dados():
    """Carrega a base bruta e aplica a limpeza idêntica à Etapa 1."""
    caminho_arquivo = 'data/raw/ProgramaRodofaunaAtropelada (1).xlsx'
    try:
        df_bruto = pd.read_excel(caminho_arquivo)
    except FileNotFoundError:
        caminho_arquivo = 'data/raw/ProgramaRodofaunaAtropelada.xlsx'
        df_bruto = pd.read_excel(caminho_arquivo)
    
    # Remoção de duplicatas técnicas estritas (ignorando OBJECTID)
    colunas_para_verificar = [col for col in df_bruto.columns if col != 'OBJECTID']
    df_unico = df_bruto.drop_duplicates(subset=colunas_para_verificar, keep='first').copy()
    
    # Filtro exclusivo para fauna silvestre com classe identificada
    df_silvestre = df_unico[df_unico['silvestre_'] == 'silvestre'].copy()
    df_silvestre = df_silvestre[df_silvestre['classe'] != 'nao_identificado'].copy()
    
    return df_silvestre

def executar_preprocessamento():
    print("Iniciando a Etapa 2 - Pré-processamento e Engenharia de Atributos...\n")
    garantir_pastas()
    
    df = carregar_e_limpar_dados()
    print(f"Base limpa carregada: {len(df)} registros de fauna silvestre.")
    
    # 1. Engenharia Temporal (Nível de Registro)
    df['data_dt'] = pd.to_datetime(df['data_dt'])
    df['ano'] = df['data_dt'].dt.year
    df['mes'] = df['data_dt'].dt.month
    df['dia'] = df['data_dt'].dt.day
    df['dia_semana_real'] = df['data_dt'].dt.day_name()
    
    # Estação climática (Cerrado/DF): Seca (Maio a Outubro) vs Chuvosa (Novembro a Abril)
    df['estacao'] = df['mes'].apply(lambda m: 'seca' if 5 <= m <= 10 else 'chuvosa')
    
    # 2. Engenharia Espacial: Discretização em Grade de 500m x 500m
    GRID_SIZE = 500
    df['grid_x'] = (df['x'] // GRID_SIZE).astype(int)
    df['grid_y'] = (df['y'] // GRID_SIZE).astype(int)
    df['cell_id'] = df['grid_x'].astype(str) + '_' + df['grid_y'].astype(str)
    
    # Salva o dataset processado no nível de ocorrência
    caminho_dados_processados = 'data/processed/dados_processados.csv'
    df.to_csv(caminho_dados_processados, index=False)
    print(f"[OK] Ficheiro '{caminho_dados_processados}' salvo ({len(df)} linhas).")
    
    # 3. Construção da Grade Espaço-Temporal Agregada por Mês (celula_mes.csv)
    celulas_unicas = df[['cell_id', 'grid_x', 'grid_y']].drop_duplicates()
    meses_anos_unicos = df[['ano', 'mes']].drop_duplicates().sort_values(by=['ano', 'mes'])
    
    # Produto cartesiano (gera células ativas x meses para inclusão de ausências)
    celulas_unicas['key'] = 1
    meses_anos_unicos['key'] = 1
    matriz_base = pd.merge(celulas_unicas, meses_anos_unicos, on='key').drop('key', axis=1)
    
    # Contagem de atropelamentos por célula/mês
    contagem_reais = df.groupby(['cell_id', 'ano', 'mes']).size().reset_index(name='contagem')
    
    # Junção da matriz com as ocorrências reais
    celula_mes = pd.merge(matriz_base, contagem_reais, on=['cell_id', 'ano', 'mes'], how='left')
    celula_mes['contagem'] = celula_mes['contagem'].fillna(0).astype(int)
    
    # Alvo Binário: 1 (ocorrência/hotspot no mês) vs 0 (ausência)
    celula_mes['atropelamento'] = (celula_mes['contagem'] > 0).astype(int)
    celula_mes['estacao'] = celula_mes['mes'].apply(lambda m: 'seca' if 5 <= m <= 10 else 'chuvosa')
    
    # Coordenadas de centroide das células (UTM)
    celula_mes['centro_x'] = celula_mes['grid_x'] * GRID_SIZE + (GRID_SIZE / 2)
    celula_mes['centro_y'] = celula_mes['grid_y'] * GRID_SIZE + (GRID_SIZE / 2)
    
    caminho_celula_mes = 'data/processed/celula_mes.csv'
    celula_mes.to_csv(caminho_celula_mes, index=False)
    
    total_presencas = celula_mes['atropelamento'].sum()
    total_ausencas = len(celula_mes) - total_presencas
    
    print(f"[OK] Ficheiro '{caminho_celula_mes}' salvo ({len(celula_mes)} linhas agregadas).")
    print(f"     • Ocorrências (atropelamento = 1): {total_presencas}")
    print(f"     • Ausências (atropelamento = 0): {total_ausencas}")
    print("\nEtapa 2 concluída com sucesso! Ficheiros salvos em 'data/processed/'.")

if __name__ == '__main__':
    executar_preprocessamento()