# 🧠 Identificação de sentimentos e emoções em textos nas redes sociais
**Link Youtube:** https://youtu.be/mxqNiozouvg

# 👥 Integrantes:
- Julian de Campos Teixeira - 10400765
- Luis Gustavo Aguirre Castanho - 10401017
- Luiz Henrique Bonilha Pasquinelli - 10401415
- Vinicius Moreira Tamburu - 10401551

## 📌 Problema

Este projeto tem como objetivo classificar automaticamente emoções e sentimentos em mensagens de redes sociais (tweets) escritas em português brasileiro. A partir da análise textual, buscamos identificar as emoções humanas expressas como:

- Raiva
- Tristeza
- Confiança
- Medo
- Amor
- Alegria
- Ausente (neutro ou sem emoção definida)

E os sentimentos Negativo, Positivo ou Neutro.

A identificação dessas emoções pode apoiar estudos sobre opinião pública, comportamento online e tendências sociais.

---

## 🛠️ Soluções Implementadas

- 🔎 **Pré-processamento**: Limpeza dos tweets com remoção de emojis, links, menções e expansão de gírias.
- 💬 **Classificação multi-rótulo**: Um mesmo tweet pode expressar mais de uma emoção.
- 🤖 **Modelagem com Multinomial Naives Bayes (MNB)**: Utilização do modelo MNB para classificação de emoções e sentimentos.
- 🤖 **Modelagem com BERTimbau**: Utilização do modelo BERTimbau pré-treinado com fine-tuning para língua portuguesa para classificação de sentimentos.
- 📊 **Métricas de avaliação**: Precisão, Recall e F1-Score para avaliação do desempenho por classe.
- 🧪 **Testes com base anotada**: Base de 200 tweets rotulados manualmente como referência de validação.

