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

A próxima fase é:

**Etapa 5 — Geração dos patches de 256×256**

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

---

# Etapa 4 — Divisão em treino, validação e teste

Código:

```text
src/data/split.py
```

A Etapa 4 dividiu as **10.000 pinturas** de `dataset_final.csv` em três conjuntos, **no nível da pintura original e antes da geração dos patches**. Cada `Painting_ID` pertence a exatamente um conjunto, e todos os patches futuros herdarão o conjunto da pintura de origem.

## Proporções

Divisão **70% / 15% / 15%**:

| Conjunto | Pinturas | Por macroclasse |
|---|---|---|
| Treino | 7.000 | 1.750 |
| Validação | 1.500 | 375 |
| Teste | 1.500 | 375 |

## Método

- Estratificação por **macroclasse** e por **estilo original**, de modo que Renaissance e Cubism preservem a proporção de seus subestilos em cada conjunto.
- Método dos maiores restos para garantir contagens exatas.
- Divisão determinística, sem `pandas.sample()` e sem bibliotecas de split.
- `SPLIT_SEED = 42`, independente da semente da Etapa 3.
- Cada pintura recebe uma chave SHA-256 calculada a partir da semente e do `Painting_ID`. As pinturas de cada estrato são ordenadas por essa chave e distribuídas entre os conjuntos.

Composição por subestilo (treino / validação / teste):

- Early Renaissance: 371 / 79 / 80;
- High Renaissance: 357 / 77 / 76;
- Mannerism / Late Renaissance: 342 / 73 / 73;
- Northern Renaissance: 680 / 146 / 146;
- Analytical Cubism: 75 / 16 / 17;
- Cubism: 1.527 / 327 / 327;
- Synthetic Cubism: 148 / 32 / 31.

## Verificações automáticas (anti-vazamento)

O script interrompe a execução com erro se qualquer verificação falhar:

- todos os `Painting_ID` são únicos;
- nenhuma sobreposição de `Painting_ID` entre os conjuntos;
- nenhum SHA-256 de pixels aparece em conjuntos diferentes;
- a soma dos conjuntos é igual a 10.000;
- as contagens por macroclasse e por conjunto são exatas;
- refazer a divisão produz exatamente o mesmo resultado.

## Distribuição de gênero, autoria e orientação

- Gênero: diferença máxima entre conjuntos inferior a 1,1 p.p. no total. Dentro de cada macroclasse, a diferença máxima foi de 2,86 p.p. (Impressionism) a 4,38 p.p. (Cubism), compatível com amostras de 375 pinturas.
- `Unknown Artist`: 51,49% no treino, 51,80% na validação e 53,13% no teste.
- Artistas identificados variaram menos de 0,6 p.p. entre conjuntos.
- Orientação vertical: 58,80% / 58,87% / 60,00%; horizontal: 41,00% / 40,93% / 39,73%.

A distribuição não foi artificialmente corrigida.

## Limitações

- A divisão **não é agrupada por artista**, pois `Unknown Artist` representa mais da metade das obras e não pode ser tratado como um único autor. Obras de um mesmo artista identificado podem aparecer em conjuntos diferentes.
- 3 rótulos de artista aparecem apenas em validação ou teste, e não no treino.
- A possível dependência de características de autores específicos deverá ser considerada na análise de *shortcut learning*.

## Como reproduzir

Na raiz do projeto:

```bash
python -m src.data.split --manifest data/metadata/dataset_final.csv --out data/metadata
```

Os arquivos `train.csv`, `validation.csv` e `test.csv` são regravados com conteúdo idêntico.

---

# Arquivos de metadados

Os principais arquivos estão em:

```text
data/metadata/
```

## `auditoria_integridade.csv`

Diagnóstico técnico das 26.426 obras candidatas.

## `candidatos_quase_duplicatas.csv`

Pares identificados pelo pHash e respectivas classificações manuais.

## `grupos_duplicatas_confirmadas.csv`

Grupos consolidados de registros que representam a mesma obra.

## `descartes_auditoria.csv`

150 registros descartados durante a auditoria.

## `dataset_final.csv`

Manifesto definitivo das 10.000 pinturas selecionadas.

## `train.csv`, `validation.csv` e `test.csv`

Manifestos da divisão da Etapa 4, com 7.000, 1.500 e 1.500 pinturas, respectivamente. Possuem as mesmas colunas de `dataset_final.csv`.

Estes são os arquivos que devem ser utilizados como entrada para as próximas etapas.

---

# Estrutura atual do projeto

```text
tcc_wikiart/
│
├── README.md
├── requirements.txt
├── .gitignore
│
├── notebooks/
│   ├── 01_analise_exploratoria.ipynb
│   ├── 02_auditoria_dataset.ipynb
│   └── 03_curadoria_dataset.ipynb
│
├── src/
│   └── data/
│       ├── labels.py
│       └── split.py
│
├── data/
│   └── metadata/
│       ├── auditoria_integridade.csv
│       ├── candidatos_quase_duplicatas.csv
│       ├── grupos_duplicatas_confirmadas.csv
│       ├── descartes_auditoria.csv
│       ├── dataset_final.csv
│       ├── train.csv
│       ├── validation.csv
│       └── test.csv
│
└── docs/
    └── project_log.md
```

Novos módulos e diretórios serão adicionados conforme o desenvolvimento avançar.

---

# Onde modificar cada parte

## Definições de classes

Alterações relacionadas às macroclasses devem ocorrer em:

```text
src/data/labels.py
```

Esse arquivo é a fonte central de definição dos rótulos.

---

## Análise exploratória

```text
notebooks/01_analise_exploratoria.ipynb
```

Utilizado para:

- análise geral do WikiArt;
- gêneros;
- artistas;
- distribuição dos estilos;
- investigação inicial de vieses.

---

## Auditoria

```text
notebooks/02_auditoria_dataset.ipynb
```

Responsável por:

- abertura das imagens;
- dimensões;
- resolução;
- hashes;
- duplicatas;
- critérios de descarte.

---

## Curadoria

```text
notebooks/03_curadoria_dataset.ipynb
```

Responsável por:

- balanceamento;
- seleção determinística;
- composição proporcional dos subestilos;
- criação do `Painting_ID`;
- incorporação de artista e gênero;
- geração de `dataset_final.csv`.

---

## Divisão dos dados

```text
src/data/split.py
```

Responsável por:

- divisão determinística em treino, validação e teste;
- estratificação por macroclasse e estilo original;
- verificações automáticas contra vazamento de dados;
- geração de `train.csv`, `validation.csv` e `test.csv`.

Os manifestos gerados estão congelados. Não altere proporções, semente ou critérios sem registrar a decisão em `docs/project_log.md` e avisar o grupo.

---

## Histórico de decisões

```text
docs/project_log.md
```

Contém o registro cronológico das principais decisões metodológicas.

Antes de modificar critérios experimentais, consulte esse arquivo.

---

# Próxima etapa — geração dos patches

A próxima etapa será a geração dos patches de `256×256` a partir das pinturas dos três manifestos.

Princípios já definidos:

- os patches são gerados a partir de `train.csv`, `validation.csv` e `test.csv`;
- todo patch herda o conjunto da pintura de origem;
- os manifestos da divisão estão congelados e não devem ser alterados.

A próxima etapa deverá definir:

- stride;
- tratamento das bordas;
- registro da posição `x/y` de cada patch;
- associação de cada patch à pintura original (`Painting_ID`);
- garantia de reconstrução posterior da pintura e dos mapas de calor;
- aplicação de *data augmentation* apenas no treino.

---

# Etapas posteriores

Após a geração dos patches:

1. definição das transformações e normalização;
2. treinamento da ResNet;
3. treinamento do Vision Transformer;
4. avaliação quantitativa;
5. agregação das previsões no nível da pintura;
6. análise de *shortcut learning*;
7. Grad-CAM;
8. demonstração final.

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

As dependências específicas de treinamento serão adicionadas quando os modelos forem implementados.

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
próxima etapa: geração dos patches
```

A execução completa das etapas anteriores não é necessariamente obrigatória para trabalhar nas etapas seguintes.

Por exemplo, se `train.csv`, `validation.csv` e `test.csv` já estiverem disponíveis e validados, a próxima fase poderá utilizá-los diretamente.

Isso evita repetir operações demoradas de auditoria e reconstrução da curadoria.

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

Esse arquivo deve ser atualizado sempre que uma decisão relevante alterar o desenho experimental do projeto.
