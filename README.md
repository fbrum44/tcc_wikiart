# Classificação de Movimentos Artísticos com Visão Computacional

Projeto de Trabalho de Conclusão de Curso voltado à classificação automática de movimentos artísticos a partir de pinturas digitalizadas utilizando técnicas de **Visão Computacional** e **Deep Learning**.

Além da classificação, o projeto pretende utilizar técnicas de **Inteligência Artificial Explicável (XAI)**, como Grad-CAM, para investigar quais regiões das imagens influenciam as decisões dos modelos.

O objetivo não é substituir a avaliação humana ou a análise de especialistas em arte, mas desenvolver uma prova de conceito capaz de:

- classificar pinturas em movimentos artísticos;
- comparar diferentes arquiteturas de Deep Learning;
- avaliar quantitativamente os modelos;
- investigar possíveis atalhos de aprendizagem (*shortcut learning*);
- utilizar técnicas de interpretabilidade para analisar as regiões visualmente relevantes para as previsões.

---

# Dataset

O projeto utiliza o dataset público **WikiArt**, disponibilizado no Hugging Face:

`huggan/wikiart`

Na versão utilizada no projeto, o conjunto possui **81.444 obras** e os seguintes atributos principais:

- `image`: imagem digital da obra;
- `artist`: artista associado à obra;
- `genre`: gênero ou temática da pintura;
- `style`: movimento ou estilo artístico.

O dataset contém **27 classes originais de estilo**.

As imagens não são armazenadas neste repositório. O dataset é carregado diretamente do Hugging Face pela biblioteca `datasets`:

```python
from datasets import load_dataset

dataset = load_dataset("huggan/wikiart")
```

Na primeira execução, o download pode demorar devido ao tamanho do conjunto. Depois disso, os arquivos permanecem no cache local do Hugging Face.

---

# Classes utilizadas no experimento

O experimento utiliza quatro macroclasses:

- Renaissance
- Baroque
- Impressionism
- Cubism

Algumas classes originais do WikiArt são agregadas para formar essas macroclasses.

## Renaissance

A classe `Renaissance` agrega:

- `Early_Renaissance`
- `High_Renaissance`
- `Northern_Renaissance`
- `Mannerism_Late_Renaissance`

Quantidade original antes da auditoria:

**6.565 imagens**

## Baroque

Utiliza diretamente:

- `Baroque`

Quantidade original:

**4.240 imagens**

## Impressionism

Utiliza diretamente:

- `Impressionism`

Quantidade original:

**13.060 imagens**

## Cubism

A classe `Cubism` agrega:

- `Cubism`
- `Analytical_Cubism`
- `Synthetic_Cubism`

Quantidade original:

**2.561 imagens**

O conjunto inicial de obras candidatas às quatro macroclasses possui:

**26.426 imagens**

As regras de agrupamento são centralizadas em:

```text
src/data/labels.py
```

Esses mapeamentos não devem ser redefinidos manualmente nos notebooks.

---

# Estado atual do projeto

Estão concluídas:

- **Etapa 1 — Análise Exploratória**
- **Etapa 2 — Auditoria de Integridade**
- **Etapa 3 — Construção do Subconjunto Final Reproduzível**
- **Etapa 4 — Divisão em treino, validação e teste**
- **Etapa 5 — Pipeline de patches**
- **Etapa 6 — Pipeline de entrada e normalização**
- **Etapa 7 — Treinamento e fine-tuning da ResNet-50**
- **Etapa 8 — Treinamento e fine-tuning do Vision Transformer (ViT-B/16)**

A próxima fase é:

**Avaliação quantitativa**

O arquivo:

```text
data/metadata/dataset_final.csv
```

é, a partir da Etapa 3, a **fonte de verdade do conjunto experimental**.

A partir da Etapa 4, os manifestos:

```text
data/metadata/train.csv
data/metadata/validation.csv
data/metadata/test.csv
```

são a **fonte de verdade da divisão** e estão **congelados**. Eles não devem ser alterados nem regenerados com outra semente ou outras proporções.

As próximas etapas não devem reconstruir a seleção a partir do WikiArt bruto, exceto quando isso for explicitamente necessário para reprodução ou auditoria.

---

# Etapa 1 — Análise Exploratória

Notebook:

```text
notebooks/01_analise_exploratoria.ipynb
```

A análise exploratória investigou:

- quantidade de imagens por estilo;
- distribuição de artistas;
- distribuição de gêneros;
- formação das macroclasses;
- presença de `Unknown Artist`;
- presença de `Unknown Genre`;
- possíveis associações entre gênero e movimento artístico;
- possíveis fontes de *shortcut learning*.

Entre os principais achados:

- Renaissance possui forte associação com `religious_painting`;
- Impressionism apresenta forte presença de `landscape`;
- Cubism apresenta concentração relevante de `still_life`;
- Baroque apresenta distribuição temática mais heterogênea;
- `Unknown Artist` não pode ser interpretado como um único autor;
- `Unknown Genre` constitui uma limitação dos metadados.

A distribuição de gênero não foi artificialmente equalizada.

---

# Etapa 2 — Auditoria de Integridade

Notebook:

```text
notebooks/02_auditoria_dataset.ipynb
```

Foram auditadas todas as **26.426 imagens candidatas**.

## Integridade técnica

Resultados:

- 26.426 imagens analisadas;
- 0 erros de abertura ou decodificação;
- 0 imagens com dimensões inválidas;
- 0 imagens com largura ou altura inferior a 256 pixels.

Todas as imagens candidatas são tecnicamente compatíveis com a estratégia de patches de `256×256`.

## Dimensões e orientação

Foram analisados:

- largura;
- altura;
- quantidade total de pixels;
- megapixels;
- proporção largura/altura;
- orientação geométrica.

Foi observada forte recorrência de dimensões envolvendo aproximadamente **1.382 pixels**, sugerindo algum tipo de padronização anterior das reproduções digitais.

Também foi identificada associação entre orientação e macroclasse:

- Impressionism apresentou predominância horizontal;
- Renaissance, Baroque e Cubism apresentaram predominância vertical.

Essa associação foi registrada como possível fonte adicional de *shortcut learning*.

---

# Auditoria de duplicatas

A redundância foi investigada em duas etapas.

## Duplicatas exatas

Foi utilizado SHA-256 calculado sobre:

- dimensões;
- pixels RGB decodificados.

Os grupos candidatos foram posteriormente confirmados por comparação pixel a pixel.

Resultado:

- 17 grupos de duplicatas exatas;
- 34 registros envolvidos;
- nenhum conflito entre macroclasses.

## Variantes digitais da mesma obra

Foi implementado um pHash de 64 bits baseado em DCT.

Foram investigados pares com distância de Hamming menor ou igual a 4.

O pHash foi utilizado somente como mecanismo de triagem. Nenhuma imagem foi descartada automaticamente com base nesse valor.

Foram encontrados:

- distância 0: 20 pares;
- distância 2: 49 pares;
- distância 4: 67 pares.

Total:

**136 pares candidatos**

Todos foram inspecionados manualmente.

Resultados:

- 135 pares confirmados como variantes da mesma obra;
- 1 falso positivo.

Após a consolidação:

- 147 grupos únicos de obras duplicadas;
- 297 registros envolvidos;
- 150 registros excedentes;
- 0 grupos envolvendo macroclasses diferentes.

Depois da auditoria:

**26.276 obras permaneceram aprovadas.**

---

# Etapa 3 — Construção do Subconjunto Final

Notebook:

```text
notebooks/03_curadoria_dataset.ipynb
```

A Etapa 3 construiu o conjunto definitivo utilizado nas próximas fases do projeto.

Após a auditoria, estavam disponíveis:

- Renaissance: 6.532;
- Baroque: 4.193;
- Impressionism: 12.993;
- Cubism: 2.558.

Foi definido um subconjunto balanceado com:

```text
2.500 pinturas por macroclasse
10.000 pinturas no total
```

---

# Seleção reproduzível

Foi definida:

```python
SEED = 42
```

A seleção final não depende de `pandas.sample()`.

Cada pintura recebe uma chave SHA-256 determinística calculada a partir da semente e do `Indice_Dataset`.

As pinturas são então ordenadas por essa chave e selecionadas deterministicamente.

Com os mesmos:

- dados de entrada;
- valores de `SEED`;
- critérios de alocação;
- código;

o mesmo conjunto de pinturas é selecionado.

---

# Preservação dos estilos originais

Renaissance e Cubism são macroclasses compostas.

A seleção não equaliza artificialmente seus subestilos.

Foi utilizado o método dos maiores restos para manter aproximadamente a composição proporcional observada após a auditoria.

## Renaissance

Composição final:

- Early Renaissance: 530;
- High Renaissance: 510;
- Mannerism / Late Renaissance: 488;
- Northern Renaissance: 972.

Total:

**2.500**

## Cubism

Composição final:

- Analytical Cubism: 108;
- Cubism: 2.181;
- Synthetic Cubism: 211.

Total:

**2.500**

Baroque e Impressionism utilizam diretamente 2.500 obras de seus respectivos estilos originais.

---

# Identificação das pinturas

Cada obra do subconjunto final possui:

- `Painting_ID`;
- `Indice_Dataset`;
- SHA-256 completo dos pixels;
- estilo original;
- macroclasse;
- artista;
- gênero;
- largura;
- altura;
- resolução;
- proporção;
- orientação.

O `Painting_ID` possui o formato:

```text
wikiart_<24 primeiros caracteres do SHA-256>
```

Exemplo:

```text
wikiart_a93f0c1b57a49fc782dd83af
```

O SHA-256 completo também permanece armazenado no manifesto.

O `Indice_Dataset` é mantido porque permite recuperar a imagem na versão utilizada do WikiArt.

---

# Manifesto final

O resultado da Etapa 3 é:

```text
data/metadata/dataset_final.csv
```

Esse arquivo contém:

**10.000 pinturas**

com:

- 2.500 por macroclasse;
- 10.000 `Painting_ID` únicos;
- 10.000 `Indice_Dataset` únicos;
- nenhuma imagem duplicada previamente descartada;
- nenhum valor ausente inesperado.

Os rótulos:

```text
Unknown Artist
Unknown Genre
```

são valores válidos existentes no WikiArt e não são tratados como valores ausentes.

---

# Distribuição de gênero no conjunto final

A seleção preservou aproximadamente os padrões encontrados na análise exploratória.

Entre os principais resultados:

## Renaissance

- `religious_painting`: 56,72%;
- `portrait`: 16,64%.

## Baroque

- `religious_painting`: 23,96%;
- `portrait`: 22,48%;
- `genre_painting`: 18,12%.

## Impressionism

- `landscape`: 33,68%;
- `genre_painting`: 20,00%.

## Cubism

- `still_life`: 25,60%;
- `portrait`: 13,80%;
- `genre_painting`: 13,00%.

A distribuição temática não foi artificialmente corrigida.

Essas associações permanecerão relevantes durante a análise de *shortcut learning*.

---

# Distribuição de autoria

O subconjunto possui **79 rótulos distintos de artista**.

`Unknown Artist` representa 5.178 das 10.000 pinturas.

Esse valor **não significa que 51,78% das obras pertençam ao mesmo artista**.

`Unknown Artist` é um rótulo que pode agregar pinturas de diversos autores reais cuja identidade não está disponível nos metadados.

Sua principal consequência metodológica é limitar análises e controles baseados em autoria.

Entre os autores conhecidos existem algumas concentrações relevantes, como:

- Rembrandt em Baroque;
- Pablo Picasso em Cubism;
- Renoir e Monet em Impressionism;
- Albrecht Dürer em Renaissance.

A seleção não foi artificialmente balanceada por artista.

A possível dependência excessiva de características específicas de determinados autores será considerada posteriormente durante a avaliação dos modelos.

# Etapa 4 — Divisão em treino, validação e teste

Código:

```text
src/data/split.py
```

A Etapa 4 dividiu as **10.000 pinturas** de `dataset_final.csv` em três conjuntos, **no nível da pintura original e antes da geração dos patches**. Cada `Painting_ID` pertence a exatamente um conjunto, e todos os patches futuros herdarão o conjunto da pintura de origem.

## Proporções

Divisão **70% / 15% / 15%**:

| Conjunto | Pinturas | Por macroclasse |
|---|---:|---:|
| Treino | 7.000 | 1.750 |
| Validação | 1.500 | 375 |
| Teste | 1.500 | 375 |

## Método

- Estratificação por **macroclasse** e por **estilo original**, de modo que Renaissance e Cubism preservem aproximadamente a proporção de seus subestilos em cada conjunto.
- Método dos maiores restos para garantir contagens exatas.
- Divisão determinística, sem `pandas.sample()` e sem bibliotecas específicas de split.
- `SPLIT_SEED = 42`, definido como parâmetro próprio da Etapa 4, embora possua o mesmo valor numérico da semente utilizada na Etapa 3.
- Cada pintura recebe uma chave SHA-256 calculada a partir da semente e do `Painting_ID`. As pinturas de cada estrato são ordenadas por essa chave e distribuídas entre os conjuntos.

Composição por subestilo (treino / validação / teste):

- Early Renaissance: 371 / 79 / 80;
- High Renaissance: 357 / 77 / 76;
- Mannerism / Late Renaissance: 342 / 73 / 73;
- Northern Renaissance: 680 / 146 / 146;
- Analytical Cubism: 75 / 16 / 17;
- Cubism: 1.527 / 327 / 327;
- Synthetic Cubism: 148 / 32 / 31.

## Verificações automáticas contra vazamento

O script interrompe a execução com erro se qualquer verificação falhar.

São verificadas:

- unicidade de `Painting_ID`;
- unicidade de `Indice_Dataset`;
- unicidade de `Hash_SHA256_Pixels`;
- ausência de sobreposição de `Painting_ID` entre os conjuntos;
- ausência de sobreposição de `Indice_Dataset` entre os conjuntos;
- ausência do mesmo SHA-256 de pixels em conjuntos diferentes;
- soma dos três conjuntos igual ao conjunto original de 10.000 pinturas;
- contagens exatas por macroclasse e conjunto;
- reprodutibilidade da divisão ao executar novamente o algoritmo com os mesmos parâmetros.

A auditoria confirmou **zero sobreposição** entre treino, validação e teste pelos três identificadores utilizados.

## Distribuição de gênero, autoria e orientação

A distribuição de gênero não foi utilizada como critério adicional de estratificação, mas foi verificada após a divisão.

Considerando as distribuições globais de gênero, a maior diferença entre os três conjuntos ficou abaixo de **1,1 ponto percentual**.

Dentro de cada macroclasse, a maior diferença observada entre os splits variou de:

- **2,86 p.p.** em Impressionism;
- até **4,38 p.p.** em Cubism.

Além disso, a comparação de cada split com a distribuição da respectiva macroclasse no conjunto completo apresentou maior desvio pontual de **3,47 p.p.**, em `Cubism / validation`, para `still_life`.

Essas diferenças foram consideradas compatíveis com subconjuntos de apenas 375 pinturas por macroclasse em validação e teste, não justificando uma estratificação adicional por gênero.

Em relação à autoria:

- `Unknown Artist`: 51,49% no treino, 51,80% na validação e 53,13% no teste;
- entre os artistas identificados, a maior diferença de participação entre conjuntos foi de aproximadamente 0,55 p.p.

Em relação à orientação:

- vertical: 58,80% / 58,87% / 60,00%;
- horizontal: 41,00% / 40,93% / 39,73%;
- quadrada: 0,20% / 0,20% / 0,27%.

As distribuições não foram artificialmente corrigidas.

## Limitações

- A divisão **não é agrupada por artista**, pois `Unknown Artist` representa mais da metade das obras e não pode ser tratado como um único autor. Obras de um mesmo artista identificado podem aparecer em conjuntos diferentes.
- Três rótulos de artistas identificados aparecem em validação e/ou teste, mas não no treino:
  - `ferdinand-hodler`;
  - `konstantin-makovsky`;
  - `theo-van-rysselberghe`.
- A possível dependência de características de autores específicos deverá ser considerada posteriormente na análise de *shortcut learning*.

## Como reproduzir

Na raiz do projeto:

```bash
python -m src.data.split --manifest data/metadata/dataset_final.csv --out data/metadata
```

Com o mesmo `dataset_final.csv`, código e parâmetros, os arquivos:

```text
data/metadata/train.csv
data/metadata/validation.csv
data/metadata/test.csv
```

são gerados deterministicamente com o mesmo conteúdo.

Os três manifestos são considerados **congelados** para as próximas etapas. Alterações nas proporções, semente ou critérios de divisão devem ser tratadas como uma nova decisão experimental e registradas em `docs/project_log.md`.

---

## Histórico de decisões

```text
docs/project_log.md
```

Contém o registro cronológico das principais decisões metodológicas.

Antes de modificar critérios experimentais, consulte esse arquivo.

---

# Etapa 5 — Pipeline de patches

A Etapa 5 implementou a geração determinística da grade de patches utilizada pelos modelos.

Código:

```text id="ds118j"
src/data/patches.py
src/data/augmentations.py
```

## Configuração

Foram definidas as seguintes regras:

- patch: **256×256 pixels**;
- stride: **128 pixels**;
- sobreposição nominal: **50%**;
- nenhum resize da pintura;
- nenhum padding;
- nenhuma região periférica descartada;
- registro das coordenadas `x0`, `y0`, `x1` e `y1`;
- associação de todos os patches ao `Painting_ID` de origem.

Nas bordas, o último patch é ajustado para terminar exatamente no limite da pintura.

Quando esse ajuste exigiria deslocamento inferior a 64 pixels, o último patch regular é reposicionado em vez de adicionar um patch quase duplicado.

## Quantidade de patches

Foram gerados:

| Conjunto | Pinturas | Patches |
|---|---:|---:|
| Treino | 7.000 | 980.730 |
| Validação | 1.500 | 209.730 |
| Teste | 1.500 | 209.030 |
| **Total** | **10.000** | **1.399.490** |

Os manifestos são:

```text id="9hodl9"
data/metadata/patches_train.csv.gz
data/metadata/patches_validation.csv.gz
data/metadata/patches_test.csv.gz
data/metadata/patches_resumo.csv
```

Cada patch registra:

- `Patch_ID`;
- `Painting_ID`;
- `Indice_Dataset`;
- split;
- macroclasse;
- estilo original;
- `x0`;
- `y0`;
- `x1`;
- `y1`.

## Validação

Os três manifestos foram relidos após a gravação e passaram por verificações automáticas.

Foram confirmados:

- `Patch_ID` único;
- associação válida à pintura original;
- pertencimento ao split correto;
- dimensão exata de 256×256;
- ausência de coordenadas inválidas;
- ausência de patches ultrapassando a pintura;
- cobertura integral das bordas;
- presença de patches para todas as 10.000 pinturas.

Resultados:

```text id="22bdeo"
train:      980.730 patches validados
validation: 209.730 patches validados
test:       209.030 patches validados
```

## Reconstrução

Foi realizada uma reconstrução completa de uma pintura utilizando todos os seus patches e respectivas coordenadas.

Resultado:

```text id="fk1g5p"
Reconstrução idêntica: True
Maior diferença de pixel: 0
Pixels diferentes: 0
```

Isso confirma que a geometria dos patches permite reconstrução espacial integral.

A mesma estratégia será utilizada posteriormente para posicionar e combinar mapas Grad-CAM sobre a pintura original.

## Quantidade de patches por classe

A quantidade de pinturas permanece balanceada, mas classes com imagens maiores produzem mais patches.

No treino:

```text id="je8lvw"
Baroque         241.050
Cubism          239.100
Impressionism   241.650
Renaissance     258.930
```

Os patches não foram descartados para balancear artificialmente essas quantidades.

Esse efeito será tratado posteriormente pela estratégia de amostragem no treinamento.

## Data augmentation

O augmentation é aplicado dinamicamente **somente ao conjunto de treino**.

Transformações atuais:

- flip horizontal com probabilidade de 50%;
- rotação entre -10° e +10° com probabilidade de 50%;
- `BORDER_REFLECT_101` durante a rotação.

Não são aplicadas alterações de cor, saturação, blur, ruído, flip vertical ou rotações extremas.

Validação e teste permanecem pixel a pixel inalterados.

As imagens aumentadas não são armazenadas em disco.

---

# Etapa 6 — Pipeline de entrada e normalização

A Etapa 6 implementou o pipeline responsável por transformar os patches da Etapa 5 em entradas utilizáveis pelos modelos de aprendizado profundo.

Código principal:

```text
src/data/normalization.py
src/data/patch_dataset.py
src/data/visual_check.py
tests/test_input_pipeline.py
```

## Normalização

Os patches são originalmente representados como arrays RGB `uint8` no formato:

```text
(H, W, C) = (256, 256, 3)
```

O pipeline:

1. converte os valores de `0–255` para `0–1`;
2. aplica média e desvio padrão configuráveis;
3. reorganiza a imagem para o formato `CHW`;
4. converte o resultado para `float32`;
5. opcionalmente converte o array para `torch.Tensor`.

O preset padrão utiliza os valores tradicionais do ImageNet:

```text
mean = (0.485, 0.456, 0.406)
std  = (0.229, 0.224, 0.225)
```

A normalização permanece configurável, pois sua compatibilidade deverá ser confirmada com os pesos pré-treinados específicos escolhidos nas etapas de implementação dos modelos.

## Dataset de patches

Foi implementado `PatchDataset`, responsável por:

```text
manifesto de patches
        ↓
Indice_Dataset
        ↓
carregamento da pintura WikiArt
        ↓
recorte pelas coordenadas x/y
        ↓
augmentation, somente no treino
        ↓
normalização
        ↓
CHW float32
        ↓
torch.Tensor
```

Os patches não são armazenados individualmente em disco.

A imagem original é recuperada a partir do WikiArt e o patch é extraído dinamicamente utilizando as coordenadas registradas na Etapa 5.

Cada item mantém informações necessárias para rastreabilidade posterior:

- `label`;
- `Patch_ID`;
- `Painting_ID`;
- `x0`;
- `y0`.

## Classes

A ordem oficial das classes continua centralizada em:

```text
src/data/labels.py
```

`patch_dataset.py` utiliza diretamente `CLASSES_FINAIS`, evitando definições independentes de índices de classe em diferentes partes do projeto.

## Data augmentation

As transformações implementadas na Etapa 5 foram integradas ao pipeline real de treino.

Somente o conjunto de treino pode receber augmentation:

- flip horizontal com probabilidade de 50%;
- rotação entre -10° e +10° com probabilidade de 50%.

Validação e teste não recebem augmentation.

O pipeline também foi ajustado para que o gerador pseudoaleatório utilizado pelo augmentation seja controlado pelas sementes do processo de treinamento, aumentando a reprodutibilidade entre execuções.

Não são aplicadas alterações cromáticas agressivas.

## DataLoaders

Foram implementados DataLoaders para:

```text
train
validation
test
```

Validação e teste utilizam todos os seus patches em ordem determinística e sem augmentation.

Para o treino foi implementado `PaintingGroupedSampler`.

Como pinturas maiores produzem mais patches, o sampler permite selecionar a mesma quantidade de patches por pintura em cada época, evitando que uma pintura tenha maior peso apenas por possuir dimensões maiores.

O número de patches por pintura é configurável.

O valor padrão atual é:

```text
patches_per_painting = 8
```

Esse valor ainda não é considerado um hiperparâmetro experimental definitivo e poderá ser revisto durante a Etapa 7.

## Reprodutibilidade

A geração dos batches utiliza sementes explícitas.

O pipeline controla:

- sorteio das pinturas;
- sorteio dos patches por pintura;
- seeds dos workers do DataLoader;
- operações aleatórias do augmentation.

Isso permite repetir uma execução com a mesma configuração experimental e obter o mesmo comportamento pseudoaleatório inicial.

## Testes automatizados

Foi criada:

```text
tests/test_input_pipeline.py
```

A suíte final contém **16 testes**, todos concluídos com sucesso.

Entre as verificações estão:

- normalização e desnormalização;
- valores conhecidos da normalização;
- correspondência entre patch e manifesto;
- ausência de augmentation em validation/test;
- funcionamento do augmentation real;
- rejeição de patches inválidos;
- balanceamento do sampler por pintura;
- reprodutibilidade do sampler;
- leitura dos manifestos comprimidos;
- unicidade dos identificadores;
- rejeição de macroclasses inválidas;
- uso da fonte central de classes;
- validação de parâmetros dos DataLoaders;
- geração de batches reais no formato esperado;
- reprodutibilidade dos batches de treino;
- geração das imagens de validação visual.

Resultado:

```text
16 testes passaram
```

## Validação visual

A inspeção final foi executada com:

```bash
python -m src.data.visual_check
```

Foram gerados arquivos temporários em:

```text
outputs/visual_check/
```

Esses arquivos são ignorados pelo Git.

Resultados observados:

```text
validation: maior diferença de pixel = 0
test:       maior diferença de pixel = 0
```

As amostras de validation e test foram visualmente idênticas aos patches originais.

No treino foram observadas apenas transformações esperadas do augmentation, incluindo espelhamento horizontal em algumas amostras.

Não foram observadas:

- alterações cromáticas inesperadas;
- deformações;
- bordas pretas;
- mudanças de dimensão;
- transformações em validation/test.

A Etapa 6 foi considerada validada.

---

# Estado atual do projeto

Estão concluídas:

- **Etapa 1 — Análise Exploratória**
- **Etapa 2 — Auditoria de Integridade**
- **Etapa 3 — Construção do Subconjunto Final Reproduzível**
- **Etapa 4 — Divisão em treino, validação e teste**
- **Etapa 5 — Pipeline de patches**
- **Etapa 6 — Pipeline de entrada e normalização**
- **Etapa 7 — Treinamento e fine-tuning da ResNet-50**
- **Etapa 8 — Treinamento e fine-tuning do Vision Transformer (ViT-B/16)**

A próxima fase é:

**Avaliação quantitativa**

O conjunto experimental permanece definido por:

```text
data/metadata/dataset_final.csv
```

e pelos splits congelados:

```text
data/metadata/train.csv
data/metadata/validation.csv
data/metadata/test.csv
```

Os modelos utilizam os manifestos de patches:

```text
data/metadata/patches_train.csv.gz
data/metadata/patches_validation.csv.gz
data/metadata/patches_test.csv.gz
```

O conjunto de teste permanece reservado para a etapa posterior de avaliação quantitativa final.

---

# Etapa 7 — ResNet-50

A primeira arquitetura principal do projeto foi implementada utilizando **ResNet-50** com pesos pré-treinados:

```text
ResNet50_Weights.IMAGENET1K_V2
```

Código principal:

```text
src/models/resnet.py
src/training/train_resnet.py
src/training/train_resnet_head.py
src/training/finetune_resnet.py
```

A camada classificadora original foi substituída por uma camada com quatro saídas correspondentes às macroclasses oficiais.

Os patches são fornecidos diretamente em resolução:

```text
256 × 256
```

e utilizam a normalização compatível com os pesos ImageNet escolhidos.

## Estratégia de treinamento

O treinamento foi dividido em:

```text
3 épocas — cabeça classificadora
3 épocas — fine-tuning completo
```

Configuração principal:

```text
seed: 42
batch size: 32
patches por pintura: 8
optimizer: AdamW
weight decay: 1e-4
```

Na fase da cabeça:

```text
learning rate: 1e-3
```

No fine-tuning:

```text
backbone learning rate: 1e-5
fc learning rate:       1e-4
```

As estatísticas de BatchNorm permaneceram congeladas durante o fine-tuning.

## Resultados

Melhor resultado da cabeça classificadora:

```text
Validation loss: 0,7550
Validation accuracy: 69,93%
```

Melhor resultado após fine-tuning:

```text
Validation loss: 0,6789
Validation accuracy: 72,43%
```

O fine-tuning aumentou a accuracy de validação em aproximadamente:

```text
+2,49 pontos percentuais
```

O melhor checkpoint da arquitetura é:

```text
models/resnet50_finetune_best.pt
```

Os checkpoints não são armazenados no GitHub.

Resultados pequenos e reproduzíveis são armazenados em:

```text
outputs/resnet50/
```

incluindo históricos, configurações e curvas de treinamento.

As métricas atuais são calculadas no nível de **patches**. A avaliação final no nível da pintura e o conjunto de teste serão tratados posteriormente.

---

# Execução da ResNet no Google Colab

Foi criado um notebook específico para reproduzir e operar a Etapa 7:

```text
notebooks/04_treinamento_resnet_colab.ipynb
```

O treinamento oficial foi realizado no Google Colab utilizando GPU **Tesla T4**.

O notebook organiza:

```text
GPU / CUDA
    ↓
clone do GitHub
    ↓
dependências
    ↓
cache do WikiArt
    ↓
Google Drive compartilhado
    ↓
checkpoints
    ↓
smoke tests opcionais
    ↓
treinamento da cabeça
    ↓
fine-tuning
    ↓
curvas e resultados
```

O WikiArt é mantido temporariamente em:

```text
/content/hf_cache
```

Como a VM do Colab é temporária, esse cache precisa ser reconstruído em uma nova VM.

Os checkpoints do grupo ficam em uma pasta compartilhada do Google Drive:

```text
TCC_WikiArt_Compartilhado/
├── models/
└── outputs/
    ├── resnet50/
    └── vit/
```

Cada integrante adiciona um atalho dessa pasta diretamente em **Meu Drive**, permitindo que o notebook utilize o caminho comum:

```text
/content/drive/MyDrive/TCC_WikiArt_Compartilhado
```

Os arquivos `.pt` permanecem no Drive e não são versionados no GitHub.

---

# Etapa 8 — Vision Transformer (ViT-B/16)

O segundo modelo principal do projeto foi implementado com Transfer Learning, sob as mesmas condições experimentais da ResNet-50, para permitir uma comparação direta entre as arquiteturas.

Código principal:

```text
src/models/vit.py
src/training/train_vit_head.py
src/training/finetune_vit.py
```

## Modelo

Foi utilizado o ViT-B/16 da biblioteca `timm`, com pesos pré-treinados no ImageNet-21k e ajustados no ImageNet-1k:

```text
vit_base_patch16_224.augreg_in21k_ft_in1k
```

A cabeça classificadora foi substituída por uma camada com quatro saídas, correspondentes às macroclasses oficiais.

A entrada permanece em `256 × 256`: o modelo é criado com `img_size=256`, e o `timm` interpola o embedding de posição. Não há redimensionamento nem recorte.

A normalização esperada pelos pesos é `mean = std = (0.5, 0.5, 0.5)`, e não a do ImageNet usada na ResNet. O pipeline utiliza o preset `half` de `src/data/normalization.py`, e `validar_normalizacao()` confere o preset contra o checkpoint.

## Condições comparáveis à ResNet-50

```text
seed: 42
batch size: 32
patches por pintura (treino e validação): 8
optimizer: AdamW
weight decay: 1e-4
cabeça: 3 épocas, learning rate 1e-3
fine-tuning: 3 épocas, backbone 1e-5, cabeça 1e-4
critério de checkpoint: menor validation loss
```

Os scripts do ViT reutilizam o laço de treino, a validação, o amostrador fixo de validação e a gravação de histórico e de checkpoint da ResNet, de modo que as duas arquiteturas são avaliadas exatamente nos mesmos 12.000 patches de validação.

O ViT não possui BatchNorm. Na fase da cabeça, o backbone permanece em `eval()` e apenas a cabeça em `train()`.

## Resultados

Melhor resultado da cabeça classificadora (época 1):

```text
Validation loss: 0,8235
Validation accuracy: 70,48%
```

Na cabeça, a loss de validação do ViT oscilou (0,82 → 0,97 → 0,85), e o fine-tuning partiu do checkpoint da época 1, escolhido pelo critério definido.

Melhor resultado após fine-tuning (época 3 do fine-tuning):

```text
Validation loss: 0,5317
Validation accuracy: 78,52%
```

## Comparação com a ResNet-50

Métricas no nível de patches, no conjunto de validação:

| Modelo | Validation loss | Validation accuracy |
|---|---:|---:|
| ResNet-50 | 0,6789 | 72,43% |
| ViT-B/16 | 0,5317 | 78,52% |

Nas condições testadas, o ViT-B/16 obteve acurácia de validação 6,1 pontos percentuais superior à da ResNet-50. Com apenas a cabeça, os dois ficaram próximos de 70%; a diferença surge no fine-tuning.

Limitações desta comparação:

- cada modelo foi treinado uma vez, com uma única seed;
- as métricas são no nível de patches, na validação, e não representam o desempenho final no nível da pintura;
- o conjunto de teste permanece reservado e não foi utilizado;
- a comparação não explica por que o ViT acerta mais, o que depende da análise de interpretabilidade.

Os resultados pequenos e reproduzíveis estão em:

```text
outputs/vit/
```

incluindo históricos, configurações, curvas e a comparação com a ResNet. Os checkpoints (`models/vit_head_best.pt` e `models/vit_finetune_best.pt`) permanecem no Google Drive compartilhado e não são versionados.

---

# Execução do ViT no Google Colab

O treinamento do ViT é operado pelo notebook:

```text
notebooks/05_treinamento_vit_colab.ipynb
```

Ele segue a mesma estrutura do notebook da ResNet (GPU, clone do GitHub, dependências, cache do WikiArt, Google Drive compartilhado, treinamento da cabeça, fine-tuning, curvas e backup) e acrescenta uma seção final de comparação com a ResNet-50.

O treinamento oficial foi realizado em GPU **Tesla T4**. A primeira época do fine-tuning levou cerca de 52 minutos.

Observação: o script de fine-tuning registra `overall_epoch` como a época do melhor checkpoint da cabeça somada à época do fine-tuning. Como o melhor checkpoint da cabeça do ViT foi o da época 1, esses valores se sobrepõem às épocas da cabeça. As curvas consolidadas usam uma linha do tempo própria (`epoch_timeline`: cabeça nas épocas 1–3 e fine-tuning nas 4–6).

---

# Etapa 9 — Avaliação quantitativa

O conjunto experimental permanece definido por:

```text
data/metadata/dataset_final.csv
```

e pelos splits congelados:

```text
data/metadata/train.csv
data/metadata/validation.csv
data/metadata/test.csv
```

O conjunto de teste, mantido reservado durante todo o desenvolvimento e treinamento, foi utilizado pela primeira vez na Etapa 9 para a avaliação quantitativa final.

A próxima fase do projeto é a **análise de interpretabilidade**, incluindo geração de mapas de ativação e investigação de possíveis padrões de *shortcut learning*.

---


A avaliação quantitativa final foi realizada com os melhores checkpoints definidos previamente pela menor loss de validação:

```text
ResNet-50:
models/resnet50_finetune_best.pt

ViT-B/16:
models/vit_finetune_best.pt
```

Após a abertura do conjunto de teste, nenhum treinamento adicional ou alteração de hiperparâmetros foi realizado.

O teste contém:

```text
1.500 pinturas
375 pinturas por macroclasse
209.030 patches
```

Todos os patches foram avaliados sem augmentation.

## Implementação

Foram criados:

```text
src/evaluation/evaluate_model.py
src/evaluation/aggregate_painting.py
src/evaluation/compare_models.py
```

O notebook operacional da etapa é:

```text
notebooks/06_avaliacao_quantitativa_colab.ipynb
```

A avaliação foi realizada em três níveis:

```text
patch individual
        ↓
pintura por votação majoritária
        ↓
pintura por média das probabilidades
```

Na votação majoritária, um empate é resolvido utilizando a maior probabilidade média entre as classes empatadas. Essa regra foi definida antes da avaliação oficial do conjunto de teste.

## Resultados no nível de patches

| Modelo | Accuracy | Macro-Precision | Macro-Recall | Macro-F1 | Loss |
|---|---:|---:|---:|---:|---:|
| ResNet-50 | 69,99% | 69,78% | 70,19% | 69,46% | 0,7320 |
| ViT-B/16 | 76,83% | 76,65% | 76,95% | 76,56% | 0,5673 |

O ViT apresentou vantagem de aproximadamente 6,84 pontos percentuais de accuracy e 7,10 pontos percentuais de macro-F1 em relação à ResNet no nível de patches.

## Resultados no nível da pintura

| Modelo | Agregação | Accuracy | Macro-Precision | Macro-Recall | Macro-F1 |
|---|---|---:|---:|---:|---:|
| ResNet-50 | Votação majoritária | 81,33% | 81,20% | 81,33% | 80,98% |
| ResNet-50 | Média das probabilidades | **81,67%** | **81,55%** | **81,67%** | **81,28%** |
| ViT-B/16 | Votação majoritária | 87,47% | 87,37% | 87,47% | 87,33% |
| ViT-B/16 | Média das probabilidades | **88,13%** | **88,07%** | **88,13%** | **88,01%** |

A agregação por pintura aumentou substancialmente o desempenho dos dois modelos.

Com média das probabilidades:

```text
ResNet-50:
69,99% por patch → 81,67% por pintura
ganho: +11,68 p.p.

ViT-B/16:
76,83% por patch → 88,13% por pintura
ganho: +11,31 p.p.
```

Na melhor configuração de cada modelo, o ViT apresentou vantagem de:

```text
Accuracy: +6,47 pontos percentuais
Macro-F1: +6,73 pontos percentuais
```

## Métricas por macroclasse

Resultados utilizando classificação por pintura e média das probabilidades:

| Modelo | Classe | Precision | Recall | F1-score |
|---|---|---:|---:|---:|
| ResNet-50 | Renaissance | 77,52% | 71,73% | 74,52% |
| ResNet-50 | Baroque | 81,13% | 68,80% | 74,46% |
| ResNet-50 | Impressionism | 86,41% | 89,87% | 88,10% |
| ResNet-50 | Cubism | 81,12% | 96,27% | 88,05% |
| ViT-B/16 | Renaissance | 82,57% | 82,13% | 82,35% |
| ViT-B/16 | Baroque | 85,63% | 79,47% | 82,43% |
| ViT-B/16 | Impressionism | 93,99% | 91,73% | 92,85% |
| ViT-B/16 | Cubism | 90,07% | 99,20% | 94,42% |

Os dois modelos apresentaram estrutura de dificuldade semelhante:

```text
maior dificuldade:
Renaissance / Baroque

maior separabilidade:
Impressionism / Cubism
```

A principal confusão ocorreu entre Renaissance e Baroque.

Na ResNet:

```text
Renaissance → Baroque: 13,60%
Baroque → Renaissance: 19,47%
```

No ViT:

```text
Renaissance → Baroque: 11,73%
Baroque → Renaissance: 16,27%
```

O Cubismo apresentou o maior recall, especialmente no ViT:

```text
Cubism — ViT-B/16
Recall: 99,20%

372 de 375 pinturas cubistas classificadas corretamente
```

A possível relação desses erros com conteúdo visual, gênero da pintura ou outras correlações do dataset será investigada posteriormente com técnicas de interpretabilidade.

## Artefatos

Os resultados da avaliação são armazenados em:

```text
outputs/evaluation/
```

incluindo:

```text
predições por patch
predições agregadas por pintura
métricas globais
métricas por classe
matrizes de confusão
comparações ResNet × ViT
curvas comparativas
```

As predições completas também foram preservadas para permitir novas análises sem necessidade de executar novamente os modelos.

A avaliação confirma o **ViT-B/16 com agregação pela média das probabilidades** como a configuração de melhor desempenho entre as alternativas avaliadas.

A comparação deve ser interpretada como uma comparação entre os pipelines de Transfer Learning adotados: ResNet e ViT diferem também em capacidade e histórico de pré-treinamento, portanto os resultados não isolam causalmente apenas o efeito da arquitetura.

---

# Próxima etapa — Interpretabilidade

A próxima etapa investigará quais regiões das pinturas influenciam as decisões dos modelos.

Serão utilizadas técnicas de interpretabilidade para:

- gerar mapas de ativação;
- analisar exemplos de acertos e erros;
- investigar a distinção entre Renaissance e Baroque;
- examinar os padrões associados ao elevado desempenho em Cubism e Impressionism;
- buscar evidências de *shortcut learning* relacionadas a gênero da pintura, composição, autoria ou outras correlações do dataset.

Os mapas de ativação deverão ser interpretados como regiões influentes para a decisão do modelo, e não como prova de compreensão semântica ou histórico-artística.

---

# Configuração do ambiente

## 1. Clonar o repositório

```bash
git clone https://github.com/fbrum44/tcc_wikiart.git
```

Entrar na pasta:

```bash
cd tcc_wikiart
```

---

## 2. Criar ambiente virtual

No Windows:

```bash
python -m venv .venv
```

Ativar:

```bash
.venv\Scripts\activate
```

---

## 3. Instalar dependências

```bash
pip install -r requirements.txt
```

Entre as bibliotecas atualmente utilizadas estão:

- `datasets`
- `pandas`
- `numpy`
- `matplotlib`
- `pillow`
- `opencv-python`
- `jupyter`
- `ipykernel`
- `scikit-learn`

O projeto também utiliza `torch` (PyTorch, para tensores e DataLoaders), `timm` (Vision Transformer, Etapa 8) e `pytest` (testes automatizados).

---

# Ordem de execução dos notebooks e scripts

Atualmente:

```text
01_analise_exploratoria.ipynb
        ↓
02_auditoria_dataset.ipynb
        ↓
03_curadoria_dataset.ipynb
        ↓
python -m src.data.split   (Etapa 4)
        ↓
python -m src.data.patches        (Etapa 5)
        ↓
python -m src.data.augmentations  (teste da Etapa 5)
        ↓
python tests/test_input_pipeline.py
        ↓
python -m src.data.visual_check   (Etapa 6)
        ↓
04_treinamento_resnet_colab.ipynb
        ↓
src.training.train_resnet_head
        ↓
src.training.finetune_resnet
        ↓
05_treinamento_vit_colab.ipynb
        ↓
src.training.train_vit_head
        ↓
src.training.finetune_vit
        ↓
06_avaliacao_quantitativa_colab.ipynb
        ↓
src.evaluation.evaluate_model
        ↓
src.evaluation.aggregate_painting
        ↓
src.evaluation.compare_models
        ↓
próxima etapa: interpretabilidade
```

A execução completa das etapas anteriores não é necessariamente obrigatória para trabalhar nas etapas seguintes.

Por exemplo, se `train.csv`, `validation.csv` e `test.csv` já estiverem disponíveis e validados, a próxima fase poderá utilizá-los diretamente.

Isso evita repetir operações demoradas de auditoria e reconstrução da curadoria.

Para as Etapas 7 e 8, os notebooks do Google Colab (04 e 05) são os roteiros recomendados. Os scripts continuam sendo a implementação reutilizável e podem ser executados diretamente quando necessário.

---

# Reprodutibilidade

O dataset original não é armazenado no GitHub devido ao seu tamanho.

O repositório armazena artefatos necessários para reproduzir os experimentos, incluindo:

- índices das imagens utilizadas;
- hashes;
- decisões de auditoria;
- descartes;
- seleção final;
- sementes;
- divisões de treino, validação e teste;
- configurações de treinamento;
- métricas.

Os notebooks possuem validações automáticas para detectar alterações inesperadas na estrutura utilizada do WikiArt.

Os checkpoints treinados não são versionados devido ao tamanho. Eles são preservados em Google Drive compartilhado entre os integrantes. Configurações, históricos de treinamento e curvas permanecem no repositório. O cache local do WikiArt no Google Colab é temporário e deve ser reconstruído quando uma nova VM é criada.

---

# Boas práticas para contribuição

Antes de iniciar alterações:

```bash
git pull --rebase origin main
```

Evite modificar simultaneamente o mesmo notebook quando outro integrante estiver trabalhando nele.

Antes de realizar um commit:

1. execute as células alteradas;
2. verifique se o notebook continua funcionando;
3. remova outputs muito grandes;
4. confirme que arquivos pesados não serão enviados;
5. atualize `project_log.md` quando houver decisão metodológica relevante;
6. atualize o README quando uma etapa estrutural do projeto for concluída.

Os notebooks devem preferencialmente ser versionados sem grandes outputs de imagens ou tabelas.

Não adicionar ao GitHub:

- imagens originais do WikiArt;
- cache do Hugging Face;
- `.venv`;
- checkpoints de modelos;
- arquivos temporários.

---

# Fluxo de trabalho do grupo

Antes de começar uma tarefa:

1. consultar `README.md`;
2. consultar `docs/project_log.md`;
3. identificar o notebook correspondente;
4. verificar se já existe implementação relacionada em `src/`;
5. utilizar os manifestos existentes como fonte de entrada da etapa.

Se uma lógica passar a ser reutilizada em diferentes etapas, ela deve preferencialmente ser movida para `src/`.

Se uma análise for específica de uma etapa experimental, ela pode permanecer no notebook correspondente.

---

# Histórico do projeto

O histórico cronológico das decisões metodológicas e dos principais resultados está disponível em:

```text
docs/project_log.md
```

## Árvore do projeto

```text
notebooks/
├── 01_analise_exploratoria.ipynb
├── 02_auditoria_dataset.ipynb
├── 03_curadoria_dataset.ipynb
├── 04_treinamento_resnet_colab.ipynb
├── 05_treinamento_vit_colab.ipynb
└── 06_avaliacao_quantitativa_colab.ipynb

src/
├── data/
│   ├── labels.py
│   ├── split.py
│   ├── patches.py
│   ├── augmentations.py
│   ├── normalization.py
│   ├── patch_dataset.py
│   └── visual_check.py
│
├── models/
│   ├── __init__.py
│   ├── resnet.py
│   └── vit.py
│
└── training/
│    ├── __init__.py
│    ├── train_resnet.py
│    ├── train_resnet_head.py
│    ├── finetune_resnet.py
│    ├── train_vit_head.py
│    └── finetune_vit.py
└── evaluation/
    ├── __init__.py
    ├── evaluate_model.py
    ├── aggregate_painting.py
    └── compare_models.py

outputs/
├── resnet50/
│   ├── head_config.json
│   ├── head_history.csv
│   ├── finetune_config.json
│   ├── finetune_history.csv
│   ├── resnet_history.csv
│   ├── resnet_loss.png
│   └── resnet_accuracy.png
└── vit/
│    ├── head_config.json
│    ├── head_history.csv
│    ├── finetune_config.json
│    ├── finetune_history.csv
│    ├── vit_history.csv
│    ├── vit_loss.png
│    ├── vit_accuracy.png
│    ├── comparacao_resnet_vit.png
│    └── comparacao_resnet_vit.csv
└── evaluation/
    ├── resnet50/
    ├── vit_b16/
    ├── comparison_summary.csv
    ├── comparison_by_class.csv
    ├── comparison_accuracy_macro_f1.png
    ├── comparison_f1_by_class.png
    └── comparison_validation_loss.png    
```

Novos módulos e diretórios serão adicionados conforme o desenvolvimento avançar.