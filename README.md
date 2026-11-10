# Predição de Atropelamentos e Identificação de Hotspots de Fauna Silvestre

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue.svg)](https://www.python.org/)
[![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-Latest-orange.svg)](https://scikit-learn.org/)
[![PyTorch/TensorFlow](https://img.shields.io/badge/Deep%20Learning-PyTorch%2FTensorFlow-red.svg)](https://pytorch.org/)
[![Optuna](https://img.shields.io/badge/Optuna-Hyperparameter%20Optimization-blueviolet.svg)](https://optuna.org/)

Projeto prático desenvolvido para a disciplina de **Inteligência Artificial** do curso de Sistemas de Informação da **Universidade Federal de Santa Maria (UFSM)**.

---

## 📌 Visão Geral do Projeto

O objetivo deste projeto é desenvolver um pipeline completo de Machine Learning e Redes Neurais Artificiais (Multilayer Perceptron - MLP) voltado para um problema ecológico e de infraestrutura viária real: **a predição de atropelamentos de animais silvestres e a identificação de áreas críticas (hotspots)** nas rodovias do Distrito Federal.

### 🌐 Contexto e Fonte dos Dados
Os dados utilizados provêm do **Programa RodoFauna Atropelada**, uma iniciativa do **Instituto Brasília Ambiental (IBRAM/DF)**. Entre abril de 2010 e março de 2015, foram realizados 484 percursos sistemáticos cobrindo aproximadamente 55.176 km de rodovias, gerando um registro georreferenciado abrangente das ocorrências com a fauna silvestre (aves, mamíferos, répteis e anfíbios).

- **Fonte oficial dos dados GIS:** [IBRAM/DF - Programa Rodofauna Atropelada (ArcGIS)](https://www.arcgis.com/apps/mapviewer/index.html?url=https://onda.ibram.df.gov.br/server/rest/services/Monitoramento_Ambiental/Programa_Rodofauna_Atropelada/FeatureServer/2&source=sd)

---

## 🛠️ Pipeline de Desenvolvimento

O projeto contempla as seguintes etapas sequenciais de análise e modelagem:

```text
┌──────────────────┐    ┌────────────────────┐    ┌─────────────────────┐
│ 1. Exploratória  │ ──►│ 2. Preprocessamento│ ──►│ 3. Classificação    │
│    dos Dados     │    │    & Feature Eng.  │    │    (Binária/Multi)  │
└──────────────────┘    └────────────────────┘    └─────────────────────┘
                                                             │
┌──────────────────┐    ┌────────────────────┐               ▼
│ 6. Discutir      │ ◄──│ 5. Regularização   │ ◄──┌─────────────────────┐
│    Generalização │    │    & Overfitting   │    │ 4. Regressão &      │
└──────────────────┘    └────────────────────┘    │    Otimiz. Optuna   │
                                                  └─────────────────────┘
```

### 1. Análise Exploratória de Dados (EDA)
- Filtragem exclusiva para registros de **fauna silvestre**.
- Remoção de duplicatas técnicas e identificação de distribuições temporais e espaciais.
- Análise de desbalanceamento de classes taxonômicas e valores ausentes (`nao_identificado`).

### 2. Pré-processamento e Engenharia de Atributos
- **Engenharia Temporal:** Derivação de estação do ano, período seco/chuvoso e indicador de final de semana/dia útil.
- **Engenharia Espacial:** Mapeamento em grades de 500m x 500m e agregação por mês.
- **Tratamento de Variáveis:** Encoders adequados (One-Hot) e normalização (`StandardScaler`).
- **Divisão de Dados:** Avaliação de splits temporais (Treino 2010-2013 / Teste 2014-2015) para evitar vazamento de dados.

### 3. Classificação com MLP vs. Aprendizado Tradicional
- **Classificação Binária:** Célula espacial com atropelamento (1) vs sem atropelamento (0) no mês.
- **Classificação Multiclasse:** Predição do grupo taxonômico no nível do registro (Aves, Mamíferos, Répteis, Anfíbios).
- **Modelos:** MLP, Random Forest.
- **Métricas:** Accuracy, Precision, Recall, F1-Score (Macro) e ROC-AUC.

### 4. Regressão
- Predição do número de atropelamentos agregados por célula espacial e mês.
- **Modelos Comparativos:** MLP, Regressão Linear e Random Forest Regressor.
- **Métricas:** MAE, MSE, RMSE e $R^2$.

### 5. Otimização de Hiperparâmetros (Optuna)
- Ajuste fino da arquitetura da MLP e parâmetros de treino isolados no conjunto de validação.

### 6. Regularização e Análise de Overfitting
- Avaliação de Dropout, Weight Decay (L2) e Early Stopping.

---

## 📁 Estrutura do Repositório

```text
.
├── data/
│   ├── raw/                      # Dados brutos do RodoFauna (.xlsx)
│   └── processed/                # Datasets processados e agregados (.csv)
├── outputs/
│   ├── figuras/                  # Gráficos da EDA e avaliação dos modelos
│   ├── tabelas/                  # Tabelas exportadas
│   └── modelos/                  # Pesos dos modelos treinados salvos
├── src/
│   ├── eda.py                    # Script Principal da EDA
│   ├── eda_complementar.py       # Script de análise temporal e outliers
│   ├── preprocessing.py          # Engenharia de atributos e formatação
│   ├── classification.py         # Modelos de Classificação
│   ├── regression.py             # Modelos de Regressão
│   ├── optuna_tuning.py          # Otimização
│   └── regularization.py         # Análise de generalização
├── .gitignore
├── requirements.txt
└── README.md
```

---

## 🚀 Como Executar

### Pré-requisitos
Certifique-se de ter o Python 3.8+ instalado.

1. **Clonar o Repositório:**
   ```bash
   git clone https://github.com/seu-usuario/predicao-atropelamento-fauna.git
   cd predicao-atropelamento-fauna
   ```

2. **Criar e Ativar um Ambiente Virtual:**
   ```bash
   python -m venv venv
   # No Windows:
   venv\Scripts\activate
   ```

3. **Instalar as Dependências:**
   ```bash
   pip install -r requirements.txt
   ```

4. **Executar os Scripts (Ordem Sugerida):**
   ```bash
   python src/eda.py
   python src/preprocessing.py
   ```

---

## 📊 Principais Resultados

> *(A preencher após a conclusão dos experimentos)*

| Experimento | Modelo | Métrica Principal | F1-Score / $R^2$ | ROC-AUC / RMSE |
| :--- | :--- | :---: | :---: | :---: |
| Classificação Binária | MLP (Optuna) | F1-Score | - | - |
| Classificação Binária | Random Forest | F1-Score | - | - |
| Classificação Multiclasse | MLP | F1-Macro | - | - |
| Regressão | MLP Regressor | RMSE | - | - |

---

## 👨‍💻 Autores e Contribuidores

Trabalho desenvolvido para a disciplina de Inteligência Artificial da UFSM:

- **Deivid Trindade** - [GitHub](https://github.com/deividtrindade777)
- **Weslley Menezes** - [GitHub](https://github.com/WeslleyHBM)

**Docentes Responsáveis:** Prof. Rodrigo da Rosa Righi / Prof. Luis Alvaro de Lima Silva (UFSM)