import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

# 1. Configurações Iniciais e Reprodutibilidade
RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)

sns.set_theme(style="whitegrid")
plt.rcParams['figure.figsize'] = (10, 6)

def garantir_pastas():
    """Garante que as pastas de output existam antes de salvar os arquivos."""
    os.makedirs('outputs/tabelas', exist_ok=True)
    os.makedirs('outputs/figuras', exist_ok=True)

def executar_eda_inicial():
    print("Iniciando a Etapa 1 - Análise Exploratória de Dados (EDA)...\n")
    garantir_pastas()
    
    # 2. Carregamento dos Dados Brutos
    # Tenta ler com o (1) primeiro, se não achar, lê o normal
    caminho_arquivo = 'data/raw/ProgramaRodofaunaAtropelada (1).xlsx'
    
    try:
        df_bruto = pd.read_excel(caminho_arquivo)
    except FileNotFoundError:
        caminho_arquivo = 'data/raw/ProgramaRodofaunaAtropelada.xlsx'
        df_bruto = pd.read_excel(caminho_arquivo)

    print(f"Total de registros brutos carregados: {len(df_bruto)}")
    
    # 3. Remoção de Duplicatas Técnicas
    # Ignoramos a coluna 'OBJECTID' para identificar as linhas inteiramente idênticas
    colunas_para_verificar = [col for col in df_bruto.columns if col != 'OBJECTID']
    df_unico = df_bruto.drop_duplicates(subset=colunas_para_verificar, keep='first').copy()
    print(f"Total de registros únicos (todas as faunas): {len(df_unico)}")
    
    # 4. Filtragem Exclusiva para Fauna Silvestre
    df_silvestre = df_unico[df_unico['silvestre_'] == 'silvestre'].copy()
    
    # Remove a única linha com classe "nao_identificado" conforme decisão metodológica
    df_silvestre = df_silvestre[df_silvestre['classe'] != 'nao_identificado'].copy()
    print(f"Total de registros únicos de fauna silvestre com classe identificada: {len(df_silvestre)}\n")
    
    # 5. Tabelas
    distribuicao_classes = df_silvestre['classe'].value_counts()
    distribuicao_classes.to_csv('outputs/tabelas/distribuicao_classes.csv', header=['quantidade_ocorrencias'])
    
    contagem_nao_id = (df_silvestre == 'nao_identificado').sum()
    contagem_nao_id = contagem_nao_id[contagem_nao_id > 0].sort_values(ascending=False)
    contagem_nao_id.to_csv('outputs/tabelas/contagem_nao_identificado.csv', header=['quantidade_ausente'])

    # 6. Gráfico 1: Distribuição das Classes Taxonômicas
    plt.figure()
    ax = sns.countplot(data=df_silvestre, x='classe', order=distribuicao_classes.index, palette='viridis')
    plt.title('Distribuição de Atropelamentos por Classe Taxonômica', fontsize=14)
    plt.xlabel('Classe Taxonômica', fontsize=12)
    plt.ylabel('Número de Ocorrências', fontsize=12)
    # Adiciona os números em cima de cada barra
    for p in ax.patches:
        ax.annotate(f'{int(p.get_height())}', (p.get_x() + p.get_width() / 2., p.get_height()),
                    ha='center', va='center', xytext=(0, 5), textcoords='offset points')
    plt.tight_layout()
    plt.savefig('outputs/figuras/01_distribuicao_classes.png', dpi=300)
    plt.close()
    
    # 7. Gráfico 2: Distribuição Temporal por Ano
    plt.figure()
    ax = sns.countplot(data=df_silvestre, x='ano', palette='magma')
    plt.title('Atropelamentos Registrados por Ano (2010 a 2015)', fontsize=14)
    plt.xlabel('Ano', fontsize=12)
    plt.ylabel('Número de Ocorrências', fontsize=12)
    plt.tight_layout()
    plt.savefig('outputs/figuras/02_distribuicao_anual.png', dpi=300)
    plt.close()

    # 8. Gráfico 3: Distribuição Espacial (UTM)
    plt.figure()
    sns.scatterplot(data=df_silvestre, x='x', y='y', hue='classe', alpha=0.6, palette='viridis', s=20)
    plt.title('Distribuição Espacial dos Atropelamentos (Coordenadas UTM)', fontsize=14)
    plt.xlabel('Coordenada X (UTM)', fontsize=12)
    plt.ylabel('Coordenada Y (UTM)', fontsize=12)
    plt.legend(title='Classe')
    plt.tight_layout()
    plt.savefig('outputs/figuras/03_distribuicao_espacial.png', dpi=300)
    plt.close()

    print("[OK] EDA concluída. Tabelas salvas em 'outputs/tabelas/' e gráficos salvos em 'outputs/figuras/'.")
    return df_silvestre

if __name__ == "__main__":
    df = executar_eda_inicial()