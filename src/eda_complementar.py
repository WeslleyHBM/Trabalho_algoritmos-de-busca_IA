import pandas as pd
import numpy as np

def eda_complementar():
    print("Iniciando análises complementares (Datas, Correlações e Outliers)...\n")
    
    # 1. Carregamento e Preparação Básica
    caminho = 'data/raw/ProgramaRodofaunaAtropelada (1).xlsx'
    try:
        df = pd.read_excel(caminho)
    except FileNotFoundError:
        df = pd.read_excel('data/raw/ProgramaRodofaunaAtropelada.xlsx')
        
    df_unico = df.drop_duplicates(subset=[col for col in df.columns if col != 'OBJECTID']).copy()
    df_silvestre = df_unico[df_unico['silvestre_'] == 'silvestre'].copy()
    
    # 2. Análise Temporal e Dias da Semana
    df_silvestre['data_dt_convertida'] = pd.to_datetime(df_silvestre['data_dt'], errors='coerce')
    
    # O Pandas converte o dia da semana começando do 0 (segunda) até 6 (domingo)
    # A base original está em string ('seg', 'ter', etc.)
    print(f"Total de registros com dia da semana igual a 'eg_': {(df_silvestre['dia_da_sem'] == 'eg_').sum()}")
    
    fim_de_semana = df_silvestre['data_dt_convertida'].dt.dayofweek.isin([5, 6]).sum()
    print(f"Registros ocorridos em sábados ou domingos (baseado na data convertida): {fim_de_semana}")
    
    # 3. Correlações e Redundâncias
    # Extraindo mês e dia do ano para checar correlação (0.997)
    df_silvestre['mes_num'] = df_silvestre['data_dt_convertida'].dt.month
    df_silvestre['dia_do_ano'] = df_silvestre['data_dt_convertida'].dt.dayofyear
    
    corr_mes_dia = df_silvestre['mes_num'].corr(df_silvestre['dia_do_ano'])
    print(f"\nCorrelação entre 'mes_num' e 'dia_do_ano': {corr_mes_dia:.4f}")
    
    # 4. Outliers nas Coordenadas (Método IQR)
    def checar_outliers_iqr(coluna):
        Q1 = df_silvestre[coluna].quantile(0.25)
        Q3 = df_silvestre[coluna].quantile(0.75)
        IQR = Q3 - Q1
        limite_inferior = Q1 - 1.5 * IQR
        limite_superior = Q3 + 1.5 * IQR
        outliers = df_silvestre[(df_silvestre[coluna] < limite_inferior) | (df_silvestre[coluna] > limite_superior)]
        return len(outliers)

    print(f"\nOutliers em X (UTM) usando IQR: {checar_outliers_iqr('x')}")
    print(f"Outliers em Y (UTM) usando IQR: {checar_outliers_iqr('y')}")
    
    # 5. Granularidade de Espécies
    print(f"\nQuantidade de espécies distintas (especie_nm): {df_silvestre['especie_nm'].nunique()}")
    
    print("\n[OK] Análises complementares finalizadas.")

if __name__ == "__main__":
    eda_complementar()