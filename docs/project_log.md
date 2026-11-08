# Project Log

Registro cronológico das principais decisões, resultados e mudanças metodológicas do projeto.

O objetivo deste arquivo é preservar o histórico das decisões técnicas e experimentais. Descrições detalhadas das análises permanecem nos respectivos notebooks e relatórios de marco.

---

## 2026-10 — Definição do escopo experimental

- Definidas quatro macroclasses finais:
  - Renaissance;
  - Baroque;
  - Impressionism;
  - Cubism.
- Renaissance passou a agregar:
  - `Early_Renaissance`;
  - `High_Renaissance`;
  - `Northern_Renaissance`;
  - `Mannerism_Late_Renaissance`.
- Cubism passou a agregar:
  - `Cubism`;
  - `Analytical_Cubism`;
  - `Synthetic_Cubism`.
- Definido o uso de Transfer Learning.
- Definida a geração de patches de 256×256 pixels.
- Definida a comparação entre uma CNN da família ResNet e Vision Transformer.
- Grad-CAM definido como principal técnica de interpretabilidade para a CNN.
- Definido que a divisão entre treino, validação e teste deverá ocorrer no nível da pintura antes da geração dos patches.
- Definida avaliação final também no nível da pintura, agregando as previsões dos patches.

---

## 2026-10 — Etapa 1: análise exploratória do WikiArt

- Dataset `huggan/wikiart` carregado com 81.444 imagens.
- Identificadas 26.426 obras pertencentes aos estilos selecionados.
- Contagens antes da auditoria:
  - Renaissance: 6.565;
  - Baroque: 4.240;
  - Impressionism: 13.060;
  - Cubism: 2.561.
- Definida formalmente a agregação dos quatro estilos renascentistas em `Renaissance`.
- Definida formalmente a agregação das três classes cubistas em `Cubism`.
- A lógica das macroclasses foi centralizada em `src/data/labels.py`.
- Identificada forte associação entre gênero artístico e macroclasse.
- Renaissance apresentou forte concentração de `religious_painting`.
- Impressionism apresentou forte concentração de `landscape`.
- Cubism apresentou concentração relevante de `still_life`.
- Baroque apresentou distribuição temática mais heterogênea.
- `Unknown Artist` foi considerado inadequado como critério direto de concentração de autoria, pois pode agrupar diversos autores desconhecidos.
- `Unknown Genre` foi mantido e documentado como limitação dos metadados.
- Decidido não equalizar artificialmente a distribuição de gêneros entre as classes.
- Definida a possibilidade de um experimento futuro controlado por gênero, tendo `portrait` como principal candidato.
- Gênero foi registrado como possível fonte de *shortcut learning*.

Notebook oficial:

`notebooks/01_analise_exploratoria.ipynb`

---

## 2026-11-02 — Etapa 2: auditoria de integridade

### Integridade técnica

- Auditadas 26.426 imagens candidatas.
- Nenhum erro de abertura ou decodificação encontrado.
- Nenhuma dimensão inválida encontrada.
- Nenhuma imagem apresentou largura ou altura inferior a 256 pixels.
- Todas as imagens candidatas são tecnicamente compatíveis com a estratégia de patches de 256×256.

### Dimensões e orientação

- Analisadas largura, altura, quantidade de pixels, megapixels, proporção e orientação.
- Identificada forte recorrência de dimensões envolvendo aproximadamente 1.382 pixels.
- Essa recorrência sugere processamento ou padronização anterior das reproduções digitais, sem evidência suficiente para determinar sua origem.
- As quatro macroclasses apresentaram resoluções digitais médias relativamente próximas.
- Identificada associação entre orientação da imagem e macroclasse:
  - Impressionism predominantemente horizontal;
  - Renaissance, Baroque e Cubism predominantemente verticais.
- Orientação registrada como possível fonte adicional de *shortcut learning*.

### Duplicatas exatas

- Implementado SHA-256 sobre dimensões e pixels RGB decodificados.
- O hash foi utilizado como mecanismo de identificação de candidatos.
- A confirmação final foi realizada por comparação pixel a pixel.
- Encontrados 17 grupos de duplicatas exatas.
- 34 registros envolvidos.
- Nenhum grupo apresentou conflito entre macroclasses.

### Quase-duplicatas

- Implementado pHash de 64 bits baseado em DCT.
- Adotado limiar de distância de Hamming `≤ 4` apenas para triagem.
- Encontrados 136 pares candidatos não exatos:
  - distância 0: 20 pares;
  - distância 2: 49 pares;
  - distância 4: 67 pares.
- Todos os 136 pares foram inspecionados manualmente.
- Resultado:
  - 135 pares confirmados como variantes da mesma obra;
  - 1 falso positivo.
- Decidido não realizar remoção automática baseada apenas em pHash.

### Consolidação e descartes

- Duplicatas exatas e variantes digitais foram consolidadas por componentes conexos.
- Resultado final:
  - 147 grupos únicos de obras duplicadas;
  - 297 registros envolvidos;
  - 150 registros excedentes.
- Nenhum grupo consolidado envolveu mais de uma macroclasse.
- Política de preservação definida:
  1. maior quantidade total de pixels;
  2. maior largura em caso de empate;
  3. menor `Indice_Dataset` em novo empate.
- Após a auditoria permaneceram 26.276 registros aprovados.

Notebook oficial:

`notebooks/02_auditoria_dataset.ipynb`

Principais artefatos:

- `data/metadata/auditoria_integridade.csv`
- `data/metadata/candidatos_quase_duplicatas.csv`
- `data/metadata/grupos_duplicatas_confirmadas.csv`
- `data/metadata/descartes_auditoria.csv`

---

## 2026-11-03 — Etapa 3: construção do subconjunto final reproduzível

### Definição do tamanho final

Após a auditoria, permaneceram:

- Renaissance: 6.532 obras elegíveis;
- Baroque: 4.193;
- Impressionism: 12.993;
- Cubism: 2.558.

Como todas as macroclasses mantiveram pelo menos 2.500 obras válidas, foi definida a utilização de:

- 2.500 pinturas por macroclasse;
- 10.000 pinturas no total.

Cubism permanece como classe limitante e utiliza quase integralmente seu conjunto aprovado.

### Reprodutibilidade

- Definida `SEED = 42`.
- A semente possui função exclusivamente reprodutiva.
- A seleção final não utiliza `pandas.sample()`.
- Cada candidato recebe uma chave SHA-256 determinística construída a partir da semente derivada do estrato e do `Indice_Dataset`.
- As pinturas são ordenadas por essa chave e selecionadas deterministicamente.
- O `hash()` nativo do Python não é utilizado para decisões persistentes.

### Preservação dos estilos originais

- Definido que o balanceamento ocorre no nível das quatro macroclasses.
- Renaissance e Cubism preservam aproximadamente a distribuição dos estilos originais.
- Utilizado o método dos maiores restos para garantir exatamente 2.500 obras por macroclasse após a alocação proporcional.

Composição final de Renaissance:

- Early Renaissance: 530;
- High Renaissance: 510;
- Mannerism / Late Renaissance: 488;
- Northern Renaissance: 972.

Composição final de Cubism:

- Analytical Cubism: 108;
- Cubism: 2.181;
- Synthetic Cubism: 211.

- Decidido não equalizar artificialmente os subestilos.

### Identificação e metadados

Cada pintura selecionada preserva:

- `Indice_Dataset`;
- SHA-256 dos pixels auditados;
- estilo original;
- macroclasse final;
- artista;
- gênero;
- largura;
- altura;
- quantidade total de pixels;
- megapixels;
- proporção;
- orientação.

- Criado `Painting_ID` a partir dos 24 primeiros caracteres hexadecimais do SHA-256 dos pixels, prefixados por `wikiart_`.
- O SHA-256 completo permanece armazenado.
- A unicidade dos `Painting_ID` é verificada automaticamente.
- O `Indice_Dataset` permanece disponível para recuperar a obra na versão utilizada do WikiArt.

### Verificação da distribuição de gênero

A seleção final preservou aproximadamente os padrões temáticos já observados na EDA.

Principais valores no subconjunto final:

- Renaissance:
  - `religious_painting`: 56,72%;
  - `portrait`: 16,64%.
- Baroque:
  - `religious_painting`: 23,96%;
  - `portrait`: 22,48%;
  - `genre_painting`: 18,12%.
- Impressionism:
  - `landscape`: 33,68%;
  - `genre_painting`: 20,00%.
- Cubism:
  - `still_life`: 25,60%;
  - `portrait`: 13,80%;
  - `genre_painting`: 13,00%.

- A seleção não introduziu alteração temática relevante em relação ao conjunto candidato.
- Gênero continua registrado como possível fonte de *shortcut learning*.
- A distribuição não será artificialmente corrigida nesta etapa.

### Verificação da autoria

- Identificados 79 rótulos distintos de artista no subconjunto final.
- `Unknown Artist` corresponde a 5.178 das 10.000 pinturas.
- Decidido explicitamente que `Unknown Artist` não deve ser tratado como um único autor.
- Esse rótulo pode representar um número indeterminado de autores reais.
- Sua principal implicação é limitar análises e controles baseados em autoria.
- `Unknown Artist` não constitui, por si só, um atalho diretamente acessível ao modelo, pois o modelo receberá somente as imagens.

Concentrações observadas entre autores identificados incluem:

- Baroque:
  - Rembrandt: 429 obras, 17,16% da classe;
  - Peter Paul Rubens: 148.
- Cubism:
  - Pablo Picasso: 300 obras, 12,00%;
  - Juan Gris: 188;
  - Georges Braque: 153.
- Impressionism:
  - Pierre-Auguste Renoir: 287 obras, 11,48%;
  - Claude Monet: 250 obras, 10,00%.
- Renaissance:
  - Albrecht Dürer: 264 obras, 10,56%.

Rótulos distintos de autoria por macroclasse:

- Renaissance: 15;
- Baroque: 10;
- Impressionism: 50;
- Cubism: 13.

Esses números não representam necessariamente a quantidade real de autores devido à presença de `Unknown Artist`.

- Decidido não balancear o subconjunto por artista.
- Possível dependência excessiva de características específicas de autores conhecidos deverá ser considerada posteriormente na avaliação dos modelos.

### Manifesto final

- Criado `data/metadata/dataset_final.csv`.
- Manifesto contém 10.000 registros.
- 2.500 registros por macroclasse.
- 10.000 `Painting_ID` únicos.
- 10.000 `Indice_Dataset` únicos.
- Nenhum valor ausente inesperado.
- `Unknown Artist` e `Unknown Genre` são tratados como rótulos válidos do WikiArt, e não como valores nulos.
- O CSV é relido após a gravação para validar o artefato persistido.
- As imagens originais não são armazenadas no GitHub.

Notebook oficial:

`notebooks/03_curadoria_dataset.ipynb`

Artefato principal:

`data/metadata/dataset_final.csv`

---

## Estado atual do projeto

Etapas concluídas:

- [x] definição do escopo;
- [x] análise exploratória;
- [x] auditoria de integridade;
- [x] construção do subconjunto final reproduzível.

O arquivo:

`data/metadata/dataset_final.csv`

passa a ser a fonte de verdade para as próximas etapas experimentais.

---

## 2026-11-05 — Etapa 4: divisão em treino, validação e teste

### Estratégia de divisão

- A divisão foi realizada no nível da pintura original, antes da geração de patches.
- O arquivo de entrada utilizado foi:
  - `data/metadata/dataset_final.csv`
- O conjunto final possui 10.000 pinturas.
- Foi mantida a semente:
  - `SPLIT_SEED = 42`
- Foram definidas as proporções:
  - treino: 70%;
  - validação: 15%;
  - teste: 15%.

A divisão final contém:

- `train.csv`: 7.000 pinturas;
- `validation.csv`: 1.500 pinturas;
- `test.csv`: 1.500 pinturas.

### Estratificação

A divisão foi estratificada por:

- macroclasse final;
- estilo original.

Cada uma das quatro macroclasses permaneceu perfeitamente balanceada:

| Macroclasse | Treino | Validação | Teste |
|---|---:|---:|---:|
| Renaissance | 1.750 | 375 | 375 |
| Baroque | 1.750 | 375 | 375 |
| Impressionism | 1.750 | 375 | 375 |
| Cubism | 1.750 | 375 | 375 |

A composição dos estilos originais de Renaissance e Cubism também foi preservada proporcionalmente.

### Reprodutibilidade

A divisão não utiliza `pandas.sample()` nem depende de estado interno de geradores pseudoaleatórios.

Cada pintura recebe uma chave SHA-256 determinística calculada a partir de:

- `SPLIT_SEED`;
- `Painting_ID`.

A ordenação por essa chave define a posição da pintura dentro de seu estrato.

A execução repetida com os mesmos dados, parâmetros e código produz os mesmos três manifestos.

### Verificações contra data leakage

Foram implementadas verificações automáticas utilizando três identificadores independentes:

- `Painting_ID`;
- `Indice_Dataset`;
- `Hash_SHA256_Pixels`.

Resultados:

- 0 sobreposições entre treino e validação;
- 0 sobreposições entre treino e teste;
- 0 sobreposições entre validação e teste;
- 0 identificadores duplicados dentro dos subconjuntos;
- 0 conteúdos SHA-256 presentes em mais de um split.

A união dos três manifestos recompõe exatamente as 10.000 pinturas presentes em `dataset_final.csv`:

- 0 pinturas ausentes;
- 0 pinturas adicionais;
- 0 alterações nos metadados originais.

As verificações foram implementadas com exceções explícitas em vez de depender exclusivamente da instrução `assert`, evitando que as salvaguardas possam ser desativadas pela execução otimizada do Python.

### Preservação da distribuição de gênero

O gênero não foi utilizado como variável de estratificação direta.

Após a divisão, sua distribuição foi auditada dentro de cada macroclasse e comparada com a distribuição da respectiva macroclasse no conjunto completo.

Foram utilizadas duas medidas:

- `TVD_pp`: distância de variação total, em pontos percentuais;
- `max_diff_pp`: maior diferença absoluta observada em um único gênero.

Resultados:

| Macroclasse | Split | TVD (p.p.) | Maior diferença (p.p.) |
|---|---|---:|---:|
| Baroque | train | 1,26 | 0,82 |
| Baroque | validation | 4,75 | 2,71 |
| Baroque | test | 2,40 | 1,11 |
| Cubism | train | 1,62 | 0,91 |
| Cubism | validation | 6,44 | 3,47 |
| Cubism | test | 4,23 | 1,49 |
| Impressionism | train | 0,96 | 0,54 |
| Impressionism | validation | 2,36 | 1,79 |
| Impressionism | test | 2,75 | 2,32 |
| Renaissance | train | 1,50 | 0,65 |
| Renaissance | validation | 4,84 | 3,12 |
| Renaissance | test | 2,65 | 1,69 |

A maior diferença pontual observada foi de 3,47 pontos percentuais, em `Cubism / validation`, no gênero `still_life`.

Considerando o tamanho de 375 pinturas por macroclasse nos conjuntos de validação e teste, as diferenças foram consideradas pequenas e compatíveis com o objetivo de preservar, na medida do possível, a distribuição temática original.

Decidiu-se não introduzir uma segunda estratificação por gênero, pois os manifestos atuais já mantêm satisfatoriamente essa distribuição e uma estratificação excessivamente granular poderia criar estratos muito pequenos.

### Artefatos da etapa

Código:

`src/data/split.py`

Manifestos congelados:

- `data/metadata/train.csv`
- `data/metadata/validation.csv`
- `data/metadata/test.csv`

Esses três arquivos passam a ser a fonte de entrada para a geração dos patches.

### Decisão metodológica

Os manifestos da Etapa 4 são considerados congelados.

Alterações futuras em:

- proporções;
- `SPLIT_SEED`;
- critérios de estratificação;
- ou composição dos três conjuntos

devem ser tratadas como uma nova decisão experimental e registradas antes da repetição dos treinamentos.

---

## Estado atual do projeto

Etapas concluídas:

- [x] definição do escopo;
- [x] análise exploratória;
- [x] auditoria de integridade;
- [x] construção do subconjunto final reproduzível;
- [x] divisão em treino, validação e teste.

---

## Próxima etapa

## 2026-11-05 — Etapa 5: pipeline de patches

### Configuração espacial

Definidos:

- `PATCH_SIZE = 256`;
- `STRIDE = 128`;
- sobreposição nominal de 50%.

Os patches são gerados diretamente sobre a resolução original da pintura.

Não são utilizados:

- resize da pintura;
- padding;
- descarte de regiões periféricas.

### Política de bordas

O último patch de cada eixo deve terminar exatamente na borda da pintura.

Durante os testes foi identificado que simplesmente acrescentar um novo patch na borda poderia produzir regiões quase duplicadas quando o deslocamento restante fosse muito pequeno.

Foi definido:

`EDGE_REPOSITION_THRESHOLD = 64`

Quando o deslocamento final é inferior a 64 pixels, a última posição regular é reposicionada para a borda.

Caso contrário, uma nova posição é adicionada.

Essa estratégia garante cobertura integral sem introduzir pixels artificiais e reduz redundância excessiva nas extremidades.

### Coordenadas e identificação

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

O `Patch_ID` é derivado deterministicamente do `Painting_ID` e das coordenadas iniciais.

### Manifestos

Criados:

- `data/metadata/patches_train.csv.gz`;
- `data/metadata/patches_validation.csv.gz`;
- `data/metadata/patches_test.csv.gz`;
- `data/metadata/patches_resumo.csv`.

Quantidade final:

| Split | Pinturas | Patches | Média por pintura |
|---|---:|---:|---:|
| train | 7.000 | 980.730 | 140,10 |
| validation | 1.500 | 209.730 | 139,82 |
| test | 1.500 | 209.030 | 139,35 |
| **Total** | **10.000** | **1.399.490** | — |

Os manifestos comprimidos ocupam aproximadamente 11 MB no total.

### Distribuição por macroclasse

Treino:

- Baroque: 241.050;
- Cubism: 239.100;
- Impressionism: 241.650;
- Renaissance: 258.930.

Validação:

- Baroque: 51.000;
- Cubism: 51.150;
- Impressionism: 52.620;
- Renaissance: 54.960.

Teste:

- Baroque: 50.770;
- Cubism: 51.560;
- Impressionism: 51.060;
- Renaissance: 55.640.

Decidido não remover patches para equilibrar artificialmente essas quantidades.

A quantidade diferente de patches é consequência das dimensões das pinturas.

A estratégia de treinamento deverá evitar que pinturas maiores recebam peso excessivo apenas por produzirem mais patches.

### Validação dos manifestos persistidos

Os arquivos `.csv.gz` foram relidos após a gravação.

Resultados:

- `train`: 980.730 patches validados;
- `validation`: 209.730 patches validados;
- `test`: 209.030 patches validados.

Foram verificadas:

- unicidade de `Patch_ID`;
- associação válida a `Painting_ID`;
- split correto;
- dimensão 256×256;
- coordenadas válidas;
- ausência de patches fora da imagem;
- cobertura desde `x=0` e `y=0`;
- cobertura até a borda direita e inferior;
- presença de patches para todas as pinturas.

Nenhuma inconsistência foi encontrada.

### Extração com OpenCV

Implementada extração das regiões com OpenCV/NumPy.

Foram testados patches reais de uma pintura do WikiArt.

Confirmado que os recortes:

- pertenciam ao mesmo `Painting_ID`;
- pertenciam ao mesmo `Indice_Dataset`;
- correspondiam visualmente às posições registradas;
- apresentavam dimensão 256×256.

### Reconstrução

Foi testada a reconstrução integral de uma pintura a partir de seus patches.

Nas áreas sobrepostas foi utilizada soma seguida de divisão pelo número de coberturas.

Resultado:

- `Reconstrução idêntica: True`;
- `Maior diferença de pixel: 0`;
- `Pixels diferentes: 0`.

A reconstrução foi pixel a pixel idêntica à pintura original.

Concluído que as coordenadas e a política de sobreposição permitem reconstrução espacial posterior dos mapas Grad-CAM.

A validação não afirma que o Grad-CAM será semanticamente correto; ela comprova que a infraestrutura geométrica para reconstruí-lo está correta.

### Data augmentation

Criado:

`src/data/augmentations.py`

Augmentation definido somente para treino:

- flip horizontal: probabilidade de 50%;
- rotação: probabilidade de 50%;
- ângulo: entre -10° e +10°;
- bordas de rotação: `cv2.BORDER_REFLECT_101`.

Não utilizados:

- flip vertical;
- alterações de cor;
- saturação;
- blur;
- ruído;
- crop aleatório;
- rotações extremas.

Validação e teste retornam cópias pixel a pixel inalteradas.

Os testes automáticos de augmentation foram concluídos com sucesso.

Augmentation será executado dinamicamente durante treinamento e não produzirá arquivos permanentes.

### Decisões metodológicas

- todos os patches válidos são mantidos nos manifestos;
- patches vizinhos podem possuir 50% de sobreposição;
- os patches não devem ser interpretados como observações estatisticamente independentes;
- a unidade principal de separação experimental continua sendo a pintura;
- augmentation não altera os manifestos;
- validation e test nunca recebem augmentation;
- eventual balanceamento deverá ocorrer durante a amostragem de treino, e não pela exclusão de patches nesta etapa.

### Código

- `src/data/patches.py`;
- `src/data/augmentations.py`.

---


## 2026-11-06 — Etapa 6: pipeline de entrada e normalização

### Objetivo

Implementado o pipeline responsável por transformar os patches definidos na Etapa 5 em entradas utilizáveis pelos modelos de aprendizado profundo.

Foram adicionados:

- `src/data/normalization.py`;
- `src/data/patch_dataset.py`;
- `src/data/visual_check.py`;
- `tests/test_input_pipeline.py`.

### Normalização

Implementada conversão:

```text
uint8 RGB HWC
        ↓
escala 0–1
        ↓
normalização por média/desvio
        ↓
float32 CHW
        ↓
torch.Tensor
```

O preset padrão utiliza:

```text
mean = (0.485, 0.456, 0.406)
std  = (0.229, 0.224, 0.225)
```

Esses valores correspondem ao preset convencional do ImageNet.

A normalização permanece configurável e deverá ser confirmada contra os pesos pré-treinados específicos escolhidos para cada arquitetura.

### Fonte central das classes

Foi identificado durante a auditoria da etapa que `patch_dataset.py` possuía uma ordem própria de classes diferente da definida em `src/data/labels.py`.

A duplicação foi removida.

`patch_dataset.py` passou a utilizar diretamente:

```text
CLASSES_FINAIS
```

de:

```text
src/data/labels.py
```

Assim, índices de classes possuem uma única fonte de verdade no projeto.

### Dataset

Implementado `PatchDataset`.

Cada item:

1. identifica a pintura pelo `Indice_Dataset`;
2. recupera a imagem original do WikiArt;
3. extrai o patch utilizando as coordenadas do manifesto;
4. aplica augmentation caso pertença ao treino;
5. normaliza a imagem;
6. converte o resultado para tensor.

Os patches continuam sendo gerados dinamicamente e não são armazenados individualmente em disco.

Cada item mantém também:

- `Patch_ID`;
- `Painting_ID`;
- classe;
- coordenadas espaciais.

### Augmentation

O augmentation da Etapa 5 foi integrado ao Dataset.

Permanece definido:

- flip horizontal com probabilidade de 50%;
- rotação entre -10° e +10° com probabilidade de 50%;
- nenhuma alteração cromática agressiva;
- nenhum augmentation em validation ou test.

Durante a auditoria foi identificado que o uso de `np.random.default_rng()` sem seed dentro da função de augmentation poderia escapar do controle das sementes dos DataLoaders.

O pipeline foi corrigido para utilizar o estado pseudoaleatório inicializado pelos workers.

A geração das transformações passa, portanto, a ser controlável pelas seeds do experimento.

### DataLoaders

Implementados DataLoaders para:

- treino;
- validação;
- teste.

Validation e test utilizam todos os seus patches e não recebem augmentation.

Foi implementado `PaintingGroupedSampler` para o treino.

A motivação é evitar que pinturas maiores recebam maior peso no treinamento apenas por produzirem mais patches.

O sampler seleciona a mesma quantidade de patches por pintura em cada época.

O parâmetro:

```text
patches_per_painting
```

é configurável.

O valor padrão atual é 8, mas a escolha definitiva desse hiperparâmetro será tratada durante a Etapa 7.

### Reprodutibilidade

Foram adicionados controles explícitos de seed para:

- sampler;
- geradores dos DataLoaders;
- workers;
- NumPy;
- augmentation.

Foi implementado teste verificando que duas construções equivalentes do loader, utilizando a mesma seed, produzem o mesmo primeiro batch de treino.

### Testes automatizados

A suíte:

```text
tests/test_input_pipeline.py
```

foi ampliada e executada localmente.

Resultado final:

```text
16 testes passaram
```

As verificações cobrem:

- normalização;
- desnormalização;
- formato dos patches;
- correspondência com manifesto;
- classes;
- augmentation;
- proteção de validation/test;
- sampler;
- seeds;
- manifestos;
- parâmetros inválidos;
- DataLoaders;
- batches reais;
- validação visual.

### Validação visual real

Executado:

```bash
python -m src.data.visual_check
```

Foram inspecionadas amostras reais após todas as transformações.

Resultados numéricos:

```text
validation: maior diferença de pixel = 0
test:       maior diferença de pixel = 0
```

Validation e test permaneceram pixel a pixel idênticos aos recortes originais após normalização e desnormalização.

No conjunto de treino foram observadas apenas transformações esperadas, incluindo espelhamento horizontal em algumas amostras.

Não foram identificadas alterações cromáticas inesperadas, deformações ou bordas artificiais relevantes.

**Etapa 6 concluída e validada.**

---

## 2026-11-07 — Etapa 7: treinamento e fine-tuning da ResNet-50

### Objetivo

Foi implementado e treinado o primeiro modelo principal de classificação do projeto utilizando Transfer Learning com uma arquitetura convolucional da família ResNet.

A arquitetura selecionada foi a **ResNet-50**, adaptada para classificar as quatro macroclasses oficiais:

- Renaissance;
- Baroque;
- Impressionism;
- Cubism.

O conjunto de teste permaneceu intocado durante toda esta etapa.

### Arquitetura e pesos pré-treinados

Foi criada:

`src/models/resnet.py`

A arquitetura utiliza:

```text
ResNet-50
ResNet50_Weights.IMAGENET1K_V2
```

A camada classificadora original foi substituída por:

```text
Linear(2048, 4)
```

A ordem das quatro classes continua centralizada em:

```text
src/data/labels.py
```

A entrada do modelo permanece em:

```text
256 × 256 pixels
```

Não foi aplicado redimensionamento para 224×224, pois a ResNet utiliza pooling adaptativo e aceita os patches de 256×256 diretamente.

A normalização da Etapa 6 foi validada contra a normalização esperada pelos pesos `IMAGENET1K_V2`.

Parâmetros:

```text
Parâmetros totais: 23.516.228
Treináveis durante head-only: 8.196
Treináveis durante fine-tuning: 23.516.228
```

### Estratégia de treinamento

O treinamento foi dividido em duas fases:

```text
Fase 1 — treinamento apenas da cabeça classificadora
Fase 2 — fine-tuning completo da ResNet-50
```

Essa estratégia permite inicialmente utilizar o backbone pré-treinado como extrator fixo de características e, posteriormente, adaptar as representações internas à tarefa específica de classificação de movimentos artísticos.

### Ambiente de execução

O treinamento oficial foi executado no **Google Colab** com:

```text
GPU: Tesla T4
CUDA: habilitado
batch size: 32
num_workers: 2
seed: 42
```

O dataset WikiArt é carregado do Hugging Face e mantido no cache temporário da VM:

```text
/content/hf_cache
```

Esse cache não é persistente entre VMs do Colab.

Os checkpoints são armazenados no Google Drive para não depender do armazenamento temporário da VM.

Foi posteriormente definida uma pasta compartilhada para o grupo:

```text
TCC_WikiArt_Compartilhado/
├── models/
└── outputs/
    └── resnet50/
```

No Colab, o diretório local:

```text
models/
```

é associado à pasta compartilhada do Google Drive.

Os checkpoints não são versionados no GitHub devido ao tamanho dos arquivos.

### Amostragem de treino e validação

O valor definitivo utilizado durante a Etapa 7 foi:

```text
patches_per_painting = 8
```

No treino, o `PaintingGroupedSampler` seleciona oito patches de cada pintura por época.

Com 7.000 pinturas:

```text
56.000 patches de treino por época
```

Para validação foi implementado `FixedPaintingSampler`.

Esse sampler seleciona deterministicamente oito patches por pintura e mantém exatamente o mesmo subconjunto durante todas as épocas.

Com 1.500 pinturas:

```text
12.000 patches de validação por época
```

A estratégia evita que pinturas maiores recebam maior peso apenas por produzirem mais patches.

Os 209.730 patches completos de validação permanecem disponíveis para avaliações posteriores.

### BatchNorm durante o treinamento da cabeça

Durante a primeira fase, congelar apenas `requires_grad` não seria suficiente para impedir alterações nas estatísticas internas das camadas BatchNorm.

Foi implementada uma política explícita:

```text
backbone em eval()
fc em train()
```

Assim:

- os pesos do backbone permanecem congelados;
- as estatísticas de BatchNorm permanecem congeladas;
- somente a camada `fc` recebe atualizações.

### Treinamento da cabeça classificadora

Código:

```text
src/training/train_resnet.py
src/training/train_resnet_head.py
```

Configuração oficial:

```text
epochs: 3
batch size: 32
patches por pintura no treino: 8
patches por pintura na validação: 8
optimizer: AdamW
learning rate: 1e-3
weight decay: 1e-4
seed: 42
```

Resultados:

| Época | Train loss | Train accuracy | Validation loss | Validation accuracy |
|---:|---:|---:|---:|---:|
| 1 | 0,8489 | 64,77% | 0,7996 | 67,63% |
| 2 | 0,7558 | 68,91% | 0,7601 | 69,63% |
| 3 | 0,7361 | 70,15% | 0,7550 | 69,93% |

O melhor checkpoint foi obtido na terceira época:

```text
Validation loss: 0,755048
Validation accuracy: 69,93%
```

Checkpoint:

```text
models/resnet50_head_best.pt
```

### Fine-tuning

Código:

```text
src/training/finetune_resnet.py
```

O fine-tuning parte obrigatoriamente do melhor checkpoint da cabeça classificadora.

Todos os parâmetros da ResNet foram liberados para treinamento.

Foram utilizados learning rates diferenciais:

```text
backbone: 1e-5
fc:       1e-4
```

Configuração:

```text
epochs: 3
batch size: 32
optimizer: AdamW
weight decay: 1e-4
patches por pintura no treino: 8
patches por pintura na validação: 8
seed: 42
```

Durante o fine-tuning, as estatísticas das camadas BatchNorm permaneceram congeladas.

Os parâmetros affine `weight` e `bias` das BatchNorm permaneceram treináveis, enquanto `running_mean` e `running_var` não foram atualizados.

Resultados:

| Fine-tuning | Época geral | Train loss | Train accuracy | Validation loss | Validation accuracy |
|---:|---:|---:|---:|---:|---:|
| 1 | 4 | 0,6983 | 71,36% | 0,7221 | 71,34% |
| 2 | 5 | 0,6460 | 73,68% | 0,6816 | 72,32% |
| 3 | 6 | 0,6122 | 75,24% | 0,6789 | 72,43% |

O melhor resultado foi obtido na terceira época de fine-tuning:

```text
Train loss: 0,612235
Train accuracy: 75,24%
Validation loss: 0,678914
Validation accuracy: 72,43%
```

Checkpoint:

```text
models/resnet50_finetune_best.pt
```

### Comparação entre as fases

Melhor resultado com apenas a cabeça:

```text
Validation loss: 0,7550
Validation accuracy: 69,93%
```

Melhor resultado após fine-tuning:

```text
Validation loss: 0,6789
Validation accuracy: 72,43%
```

O fine-tuning produziu:

```text
+2,49 pontos percentuais de validation accuracy
aproximadamente -10,1% de validation loss
```

A terceira época ainda apresentou melhora, porém com ganho significativamente menor que o observado entre as duas primeiras épocas do fine-tuning.

Decidiu-se encerrar o treinamento nesse ponto em vez de prolongá-lo apenas devido a pequenas variações adicionais.

Não foram observados sinais fortes de sobreajuste ao final das seis épocas.

### Curvas de treinamento

Os históricos das duas fases foram consolidados em seis épocas:

```text
1–3: treinamento da cabeça
4–6: fine-tuning
```

Foram gerados:

```text
outputs/resnet50/resnet_history.csv
outputs/resnet50/resnet_loss.png
outputs/resnet50/resnet_accuracy.png
```

As curvas apresentam queda contínua da loss de treino e validação e aumento progressivo das respectivas accuracies.

### Artefatos versionados

Resultados pequenos e reproduzíveis:

```text
outputs/resnet50/head_config.json
outputs/resnet50/head_history.csv
outputs/resnet50/finetune_config.json
outputs/resnet50/finetune_history.csv
outputs/resnet50/resnet_history.csv
outputs/resnet50/resnet_loss.png
outputs/resnet50/resnet_accuracy.png
```

Checkpoints `.pt` permanecem fora do GitHub.

### Notebook de execução no Google Colab

Foi criado:

```text
notebooks/04_treinamento_resnet_colab.ipynb
```

O notebook funciona como roteiro operacional da Etapa 7 e permite:

- verificar GPU e CUDA;
- clonar o repositório;
- instalar dependências necessárias;
- preparar o cache do WikiArt;
- montar o Google Drive compartilhado;
- localizar os checkpoints;
- verificar os manifestos;
- executar smoke tests;
- reproduzir o treinamento da cabeça;
- reproduzir o fine-tuning;
- recuperar métricas dos checkpoints;
- gerar as curvas oficiais;
- realizar backup dos resultados.

As etapas demoradas permanecem protegidas por flags `RUN_*` para evitar execução acidental.

### Reprodutibilidade e limitações

A configuração experimental registra:

- seed;
- versões de Python;
- PyTorch;
- TorchVision;
- dispositivo;
- GPU;
- hiperparâmetros;
- estratégia de validação;
- normalização;
- classes;
- checkpoint de origem.

As métricas obtidas nesta etapa ainda são métricas **no nível de patches**.

Não devem ser interpretadas como desempenho final no nível da pintura.

Ainda não foram calculados nesta etapa:

- precision;
- recall;
- F1-score;
- matriz de confusão;
- métricas por macroclasse;
- agregação de previsões por pintura;
- desempenho no conjunto de teste.

O conjunto de teste permaneceu completamente reservado.

**Etapa 7 concluída.**

---

## 2026-11-08 — Etapa 8: treinamento e fine-tuning do Vision Transformer (ViT-B/16)

### Objetivo

Foi implementado e treinado o segundo modelo principal do projeto, uma arquitetura Vision Transformer com Transfer Learning, sob as mesmas condições experimentais da ResNet-50 da Etapa 7, para permitir uma comparação direta entre as duas arquiteturas.

O conjunto de teste permaneceu intocado durante toda esta etapa.

### Arquitetura e pesos pré-treinados

Foi criado:

`src/models/vit.py`

A arquitetura utiliza o ViT-B/16 da biblioteca `timm`:

```text
vit_base_patch16_224.augreg_in21k_ft_in1k
```

Pesos pré-treinados no ImageNet-21k e ajustados no ImageNet-1k (AugReg).

A cabeça classificadora original foi substituída por:

```text
Linear(768, 4)
```

A ordem das quatro classes continua centralizada em `src/data/labels.py`.

**Entrada.** O modelo foi criado com `img_size=256`. Assim, o ViT recebe diretamente os mesmos patches 256×256 da ResNet, sem redimensionar para 224×224 e sem recorte. O `timm` interpola o embedding de posição de 14×14 para 16×16 posições (256 tokens de patch mais o token de classe).

**Normalização.** Os pesos escolhidos esperam média e desvio `0,5/0,5`, e não os valores do ImageNet usados na ResNet. Foi utilizado o preset `half` de `src/data/normalization.py` (`HALF_MEAN`, `HALF_STD`), e `validar_normalizacao()` compara o preset com a configuração do checkpoint.

Com o backbone congelado, apenas os 3.076 parâmetros da cabeça (`768 × 4 + 4`) são treináveis.

### Condições experimentais comparáveis à ResNet-50

Foram mantidos, sem alteração:

```text
seed: 42
batch size: 32
patches por pintura no treino: 8
patches por pintura na validação: 8 (mesma validação fixa da ResNet)
optimizer: AdamW
weight decay: 1e-4
critério de checkpoint: menor validation loss
sem precisão mista
```

Os scripts do ViT reutilizam os componentes genéricos da ResNet (`criar_loader_validacao_fixo`, `treinar_epoca`, `validar_epoca`, gravação de histórico e de checkpoint), de modo que o laço de treino e a validação são os mesmos nos dois modelos. A validação usa o mesmo subconjunto fixo de 12.000 patches.

O ViT não possui BatchNorm. Na fase da cabeça, o backbone foi mantido em `eval()` e apenas a cabeça em `train()`, política equivalente à usada na ResNet. No fine-tuning, a política de congelar estatísticas de BatchNorm não se aplica.

### Ambiente de execução

O treinamento foi executado no Google Colab, com GPU Tesla T4, no notebook:

`notebooks/05_treinamento_vit_colab.ipynb`

A primeira época do fine-tuning levou cerca de 52 minutos. Antes do treinamento oficial, o smoke test do fine-tuning (5 batches de treino e 3 de validação) confirmou que a T4 comporta o modelo completo com batch 32.

### Treinamento da cabeça classificadora

Código: `src/training/train_vit_head.py`

Configuração: 3 épocas, learning rate `1e-3`.

| Época | Train loss | Train accuracy | Validation loss | Validation accuracy |
|---:|---:|---:|---:|---:|
| 1 | 0,8698 | 67,74% | 0,8235 | 70,48% |
| 2 | 0,8106 | 70,55% | 0,9703 | 66,67% |
| 3 | 0,8072 | 70,77% | 0,8533 | 69,78% |

A loss de treino diminuiu a cada época, mas a loss de validação piorou depois da primeira. Pelo critério definido (menor validation loss), o melhor checkpoint da cabeça foi o da **época 1**:

```text
Validation loss: 0,8235
Validation accuracy: 70,48%
```

Checkpoint: `models/vit_head_best.pt`

Essa instabilidade não ocorreu na ResNet, cuja loss de validação diminuiu nas três épocas. Uma explicação possível é que o learning rate de `1e-3` seja alto para o ViT, mas essa hipótese não foi testada. Os hiperparâmetros foram mantidos para preservar a comparabilidade.

### Fine-tuning

Código: `src/training/finetune_vit.py`

O fine-tuning parte do melhor checkpoint da cabeça (época 1). Todos os parâmetros foram liberados, com learning rates diferenciais:

```text
backbone: 1e-5
cabeça:   1e-4
```

Configuração: 3 épocas.

| Fine-tuning | Train loss | Train accuracy | Validation loss | Validation accuracy |
|---:|---:|---:|---:|---:|
| 1 | 0,6432 | 74,19% | 0,6365 | 75,27% |
| 2 | 0,5287 | 78,52% | 0,5757 | 76,81% |
| 3 | 0,4732 | 80,83% | 0,5317 | 78,52% |

O melhor resultado foi obtido na terceira época de fine-tuning:

```text
Validation loss: 0,5317
Validation accuracy: 78,52%
```

Checkpoint: `models/vit_finetune_best.pt`

O fine-tuning produziu, em relação à cabeça (época 1):

```text
+8,03 pontos percentuais de validation accuracy
aproximadamente -35,4% de validation loss
```

Ao final, o script verificou que os pesos do backbone (embedding de patches) e da cabeça foram atualizados.

A loss de validação ainda diminuía na última época, e o melhor checkpoint foi o último. O treinamento foi encerrado em 3 épocas para manter o mesmo orçamento de treino da ResNet.

### Comparação com a ResNet-50 (validação, em patches)

| Modelo | Melhor época | Validation loss | Validation accuracy |
|---|---:|---:|---:|
| ResNet-50 | 6 | 0,6789 | 72,43% |
| ViT-B/16 | 6 | 0,5317 | 78,52% |

Diferença:

```text
loss: -0,147 (aproximadamente -21,7%)
accuracy: +6,09 pontos percentuais a favor do ViT
```

Com apenas a cabeça, os dois modelos ficaram próximos de 70% de acurácia (ResNet 69,93%; ViT 70,48%). A vantagem do ViT surge no fine-tuning: ganho de cerca de 8 p.p. contra 2,49 p.p. da ResNet. Ao final, a ResNet estabilizou (validation loss 0,6816 → 0,6789 nas duas últimas épocas), enquanto a do ViT ainda caía.

Essas diferenças foram medidas em condições comparáveis, mas devem ser interpretadas com cautela (ver limitações).

### Curvas e consolidação do histórico

Foram gerados:

```text
outputs/vit/vit_history.csv
outputs/vit/vit_loss.png
outputs/vit/vit_accuracy.png
outputs/vit/comparacao_resnet_vit.png
outputs/vit/comparacao_resnet_vit.csv
```

**Nota sobre a numeração de épocas.** O script de fine-tuning calcula `overall_epoch` como a época do melhor checkpoint da cabeça somada à época do fine-tuning. Como o melhor checkpoint da cabeça foi o da época 1, os valores registrados em `finetune_history.csv` e no checkpoint são 2, 3 e 4, e coincidiriam com as épocas 2 e 3 da cabeça. A primeira versão da consolidação usou essa coluna e gerou curvas com sobreposição. Ela foi corrigida com uma linha do tempo própria (`epoch_timeline`): cabeça nas épocas 1–3 e fine-tuning nas épocas 4–6, com as duas fases desenhadas como segmentos separados e uma nota informando que o fine-tuning parte do checkpoint da época 1. Os arquivos `head_history.csv` e `finetune_history.csv` e os checkpoints não foram alterados.

### Artefatos versionados

Código:

```text
src/models/vit.py
src/training/train_vit_head.py
src/training/finetune_vit.py
notebooks/05_treinamento_vit_colab.ipynb
```

Resultados pequenos e reproduzíveis:

```text
outputs/vit/head_config.json
outputs/vit/head_history.csv
outputs/vit/finetune_config.json
outputs/vit/finetune_history.csv
outputs/vit/vit_history.csv
outputs/vit/vit_loss.png
outputs/vit/vit_accuracy.png
outputs/vit/comparacao_resnet_vit.png
outputs/vit/comparacao_resnet_vit.csv
```

Checkpoints `.pt` (`models/vit_head_best.pt` e `models/vit_finetune_best.pt`) permanecem no Google Drive compartilhado, fora do GitHub.

### Limitações

- Cada modelo foi treinado uma única vez, com uma única seed. Não foi estimada a variabilidade entre execuções.
- As métricas são **no nível de patches**, na validação. Não representam desempenho final no nível da pintura.
- A época de cada modelo foi escolhida pela loss de validação, o que torna os valores de validação levemente otimistas. A estimativa final e independente depende do conjunto de teste.
- A cabeça do ViT apresentou validação instável, e o fine-tuning partiu de um checkpoint da cabeça menos treinado que o da ResNet.
- O ViT ainda melhorava ao final do fine-tuning; o número de épocas foi fixado para manter a comparabilidade.
- A comparação mostra qual modelo acertou mais, mas não explica o motivo. Essa explicação depende da análise de interpretabilidade (Grad-CAM) e da investigação de *shortcut learning*.
- Ainda não foram calculados: precision, recall, F1-score, matriz de confusão, métricas por macroclasse, agregação das previsões por pintura e desempenho no conjunto de teste.

**Etapa 8 concluída.**

---

## Estado atual do projeto

Etapas concluídas:

- [x] definição do escopo;
- [x] análise exploratória;
- [x] auditoria de integridade;
- [x] construção do subconjunto final reproduzível;
- [x] divisão em treino, validação e teste;
- [x] pipeline de patches;
- [x] pipeline de entrada e normalização;
- [x] treinamento e fine-tuning da ResNet-50;
- [x] treinamento e fine-tuning do ViT-B/16.

Próxima etapa:

### Avaliação quantitativa

A próxima etapa avaliará os dois modelos no conjunto de teste, com métricas no nível do patch e da pintura (acurácia, precision, recall, F1-score e matriz de confusão por macroclasse). O conjunto de teste só deve ser utilizado depois que os modelos e as decisões experimentais estiverem definidos.
