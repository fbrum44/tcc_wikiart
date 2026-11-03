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

## Próxima etapa

### Etapa 4 — Divisão em treino, validação e teste

Princípio já definido:

**a divisão deve ocorrer no nível da pintura original antes da geração de patches.**

Objetivos da próxima etapa:

- definir proporções de treino, validação e teste;
- definir estratégia de estratificação;
- utilizar uma divisão reproduzível;
- garantir que cada `Painting_ID` pertença a apenas um subconjunto;
- analisar a distribuição das macroclasses após o split;
- analisar a distribuição de gêneros e artistas nos subconjuntos;
- salvar manifestos separados para treino, validação e teste;
- somente depois iniciar a geração dos patches de 256×256 pixels.

Nenhum patch deverá ser gerado antes de a divisão das pinturas estar definitivamente congelada.