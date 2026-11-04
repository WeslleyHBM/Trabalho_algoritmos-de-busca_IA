# Predição de Atropelamentos e Identificação de Hotspots de Fauna Silvestre

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue.svg)](https://www.python.org/)
[![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-Latest-orange.svg)](https://scikit-learn.org/)
[![PyTorch/TensorFlow](https://img.shields.io/badge/Deep%20Learning-PyTorch%2FTensorFlow-red.svg)](https://pytorch.org/)
[![Optuna](https://img.shields.io/badge/Optuna-Hyperparameter%20Optimization-blueviolet.svg)](https://optuna.org/)

Projeto prático desenvolvido para a disciplina de **Inteligência Artificial** do curso de Ciências da Computação / Engenharia da **Universidade Federal de Santa Maria (UFSM)**.

---

## 📌 Visão Geral do Projeto

O objetivo deste projeto é desenvolver um pipeline completo de Machine Learning e Redes Neurais Artificiais (Multilayer Perceptron - MLP) voltado para um problema ecológico e de infraestrutura viária real: **a predição de atropelamentos de animais silvestres e a identificação de áreas críticas (hotspots)** nas rodovias do Distrito Federal.

### 🌐 Contexto e Fonte dos Dados
Os dados utilizados provêm do **Programa RodoFauna Atropelada**, uma iniciativa do **Instituto Brasília Ambiental (IBRAM/DF)**. Entre abril de 2010 e março de 2015, foram realizados 484 percursos sistemáticos cobrindo aproximadamente 55.176 km de rodovias, gerando um registro georreferenciado abrangente das ocorrências com a fauna silvestre (aves, mamíferos, répteis e anfíbios).

* **Fonte oficial dos dados GIS:** [IBRAM/DF - Programa Rodofauna Atropelada (ArcGIS)](https://www.arcgis.com/apps/mapviewer/index.html?url=https://onda.ibram.df.gov.br/server/rest/services/Monitoramento_Ambiental/Programa_Rodofauna_Atropelada/FeatureServer/2&source=sd)

---

## 🛠️ Pipeline de Desenvolvimento

O projeto contempla as seguintes etapas sequenciais de análise e modelagem:

```
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
- Identificação de distribuições temporais (ano, mês, dia) e espaciais (coordenadas UTM $X, Y$).
- Análise de desbalanceamento de classes taxonômicas e valores ausentes ou desconsiderados (`nao_identificado`).

### 2. Pré-processamento e Engenharia de Atributos
- **Engenharia Temporal:** Derivação de estação do ano, período seco/chuvoso e indicador de final de semana/dia útil a partir de `data_dt`.
- **Engenharia Espacial:** Mapeamento de segmentos rodoviários, distância entre ocorrências e densidade espacial de atropelamentos.
- **Tratamento de Variáveis:** Encoders adequados (One-Hot / Ordinal Encoding) para categóricas e dimensionamento de escalas (**StandardScaler / MinMaxScaler / RobustScaler**) imprescindíveis para redes MLP.
- **Divisão de Dados Espaço-Temporal:** Avaliação e discussão de *splits* aleatórios vs. *splits* temporais/espaciais para evitar vazamento de dados (*data leakage*) de pontos vizinhos.

### 3. Classificação com MLP vs. Aprendizado Tradicional
- **Classificação Binária:** Classificação de segmentos rodoviários em *Alto Risco/Ocorrência (1)* vs. *Baixo Risco/Ausência (0)*.
- **Classificação Multiclasse:** Predição do grupo taxonômico (*Aves, Mamíferos, Répteis, Anfíbios*).
- **Modelos Comparativos:** Random Forest, Logistic Regression e Gradient Boosting.
- **Métricas:** *Accuracy, Precision, Recall, F1-Score (Macro) e ROC-AUC*.

### 4. Regressão
- Predição do número total ou taxa de atropelamentos agregados por unidade espacial/temporal (ex: segmento rodoviário por mês).
- **Modelos Comparativos:** Regressão Linear, Random Forest Regressor e Gradient Boosting Regressor.
- **Métricas:** *MAE, MSE, RMSE e $R^2$*.

### 5. Otimização de Hiperparâmetros (Optuna)
- Ajuste fino da arquitetura da MLP (número de camadas, neurônios, funções de ativação) e hiperparâmetros de treino (*Learning Rate, Batch Size, Optimizer*).
- Execução isolada em conjunto de Validação/Treino (sem tocar no conjunto de teste final).

### 6. Regularização e Análise de Overfitting
- Avaliação de estratégias contra *overfitting*: **Dropout**, **Weight Decay ($L2$)** e **Early Stopping**.
- Comparação de curvas de *Loss* e métricas do conjunto de Treino vs. Validação.

---

## 📁 Estrutura do Repositório

```text
.
├── data/
│   ├── raw/                      # Dados brutos do RodoFauna
│   └── processed/                # Datasets processados e agregados
├── notebooks/
│   ├── 01_eda.ipynb              # Análise Exploratória dos Dados
│   ├── 02_preprocessing.ipynb    # Engenharia de atributos e pré-processamento
│   ├── 03_classification.ipynb   # Experimentos de Classificação (MLP + ML Tradicional)
│   ├── 04_regression.ipynb       # Experimentos de Regressão
│   ├── 05_optuna_tuning.ipynb    # Otimização de hiperparâmetros com Optuna
│   └── 06_regularization.ipynb   # Estudo de overfitting e generalização
├── src/
│   ├── utils.py                  # Funções utilitárias de plotagem e métricas
│   └── models.py                 # Definição das arquiteturas PyTorch / TensorFlow
├── reports/
│   └── relatorio_tecnico.pdf     # Relatório técnico final em formato acadêmico
├── requirements.txt              # Dependências do projeto
└── README.md                     # Documentação do repositório
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
   # No Linux/macOS:
   source venv/bin/activate
   # No Windows:
   venv\Scripts\activate
   ```

3. **Instalar as Dependências:**
   ```bash
   pip install -r requirements.txt
   ```

4. **Executar os Notebooks / Scripts:**
   Você pode rodar os notebooks via Jupyter ou VS Code:
   ```bash
   jupyter lab
   ```

---

## 📊 Principais Resultados

> *Preencha esta seção com a tabela comparativa dos seus melhores modelos ao finalizar os experimentos.*

| Experimento | Modelo | Métrica Principal | F1-Score / $R^2$ | ROC-AUC / RMSE |
| :--- | :--- | :---: | :---: | :---: |
| Classificação Binária | MLP (Optuna) | F1-Score | - | - |
| Classificação Binária | Random Forest | F1-Score | - | - |
| Classificação Multiclasse | MLP | F1-Macro | - | - |
| Regressão | MLP Regressor | RMSE | - | - |

---

## 👨‍💻 Autores e Contribuidores

Trabalho desenvolvido para a disciplina de Inteligência Artificial da UFSM pelos alunos:

- **Weslley Menezes** - [GitHub]([https://github.com/](https://github.com/WeslleyHBM)) - *weslley.menezes@acad.ufsm.br*
- **Deivid Trindade** - [GitHub]([https://github.com/](https://github.com/deividtrindade777)) - *deivid.trindade@acad.ufsm.br*

**Docentes Responsáveis:** Prof. Rodrigo da Rosa Righi / Prof. Luis Alvaro de Lima Silva (UFSM)

---

## 📜 Licença

Este projeto é voltado para fins puramente acadêmicos. Os dados pertencem ao Instituto Brasília Ambiental (IBRAM/DF).
