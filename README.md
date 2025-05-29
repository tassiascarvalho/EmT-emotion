# EmT EEG Emotion Recognition – Kaggle Adaptation

Este repositório contém a adaptação prática da arquitetura **EmT: A Novel Transformer for Generalized Cross-subject EEG Emotion Recognition** para aplicação sobre o dataset *EEG Brainwave Dataset: Feeling Emotions*, disponível no Kaggle. O foco deste projeto está na classificação emocional baseada em sinais EEG com validação entre sujeitos (cross-subject), utilizando transformadores e redes em grafos.

## 🔍 Objetivo

Reproduzir e adaptar o modelo **EmT** proposto por Ding et al. (2024), originalmente aplicado a bases como SEED e MAHNOB-HCI, para um dataset alternativo disponível em CSV. A arquitetura do modelo foi mantida, mas o pipeline de dados foi modificado para se adequar à estrutura do novo conjunto de dados.

## 🧩 Componentes principais

- `EMOTION.py`: Define a classe `EMOTION`, que herda de `PrepareData` e contém os métodos para:
  - Leitura do arquivo CSV (`emotions.csv`)
  - Codificação dos rótulos emocionais
  - Segmentação e transformação dos dados em janelas fixas
  - Simulação de múltiplos sujeitos com adição de ruído
  - Extração de características espectrais (rPSD)

- `main_emotion.py`: Script principal de treino e avaliação do modelo EmT adaptado, com suporte a GPU, validação cruzada (LOSO) e monitorização de métricas como acurácia e F1-score.

## 🧪 Dataset utilizado

- Kaggle: [EEG Brainwave Dataset: Feeling Emotions](https://www.kaggle.com/datasets/debarshichanda/eeg-brainwave-dataset-feeling-emotions)
- Total de amostras: 2132
- Classes: POSITIVE, NEGATIVE, NEUTRAL
- Características extraídas via FFT, PSD e rPSD, não separadas por canal
- Pré-processamento adaptado para simular validação entre sujeitos

## 📚 Base científica

O código foi desenvolvido com base na arquitetura proposta no seguinte artigo:

> Y. Ding, C. Tong, S. Zhang, M. Jiang, Y. Li, K. L. J. Liang, and C. Guan, “*EmT: A Novel Transformer for Generalized Cross-subject EEG Emotion Recognition*,” arXiv preprint arXiv:2406.18345, 2024. [Online]. Available: https://arxiv.org/abs/2406.18345

## 📊 Resultados

Os testes foram realizados com dois sujeitos simulados e ruído artificial. Os principais resultados médios foram:

- Acurácia de teste: **62.5%**
- F1-score de teste: **58.5%**
- Melhor acurácia individual (época 20): **81.8%**
- Melhor F1-score individual (época 20): **80.4%**
t=rPSD --subjects=2 --LS=1 --LS-rate=0.1

## 📎 Observações

- A estrutura original do modelo foi mantida, mas **não foi possível implementar o módulo STA e a segmentação em subsegmentos**, devido à estrutura do dataset. Isso pode ter impactado a coerência temporal do modelo.
- A simulação de sujeitos foi realizada com adição de ruído gaussiano leve para permitir validação LOSO.
