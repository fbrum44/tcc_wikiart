# Classificação de Movimentos Artísticos com Visão Computacional

Projeto de Trabalho de Conclusão de Curso voltado à classificação automática de movimentos artísticos a partir de pinturas digitalizadas utilizando técnicas de **Visão Computacional** e **Deep Learning**.

Além da classificação, o projeto utiliza técnicas de **Inteligência Artificial Explicável (XAI)** para investigar quais regiões das imagens influenciam as decisões dos modelos.

O objetivo não é substituir a avaliação humana ou a análise de especialistas em arte, mas desenvolver uma prova de conceito capaz de:

- classificar pinturas em movimentos artísticos;
- comparar diferentes arquiteturas de Deep Learning;
- avaliar quantitativamente os modelos;
- investigar possíveis atalhos de aprendizagem (*shortcut learning*);
- utilizar técnicas como **Grad-CAM** para interpretar as regiões visualmente relevantes para as previsões.

---

## Dataset

O projeto utiliza o dataset público **WikiArt**, disponibilizado no Hugging Face:

`huggan/wikiart`

Na versão utilizada no projeto, o conjunto possui **81.444 obras** e os seguintes atributos:

- `image`: imagem digital da obra;
- `artist`: artista associado à obra;
- `genre`: gênero ou temática da pintura;
- `style`: movimento ou estilo artístico.

O dataset possui **27 classes originais de estilo**.

As imagens não são armazenadas neste repositório. O dataset é carregado diretamente do Hugging Face por meio da biblioteca `datasets`:

```python
from datasets import load_dataset

dataset = load_dataset("huggan/wikiart")
```

Na primeira execução, o download pode demorar devido ao tamanho do conjunto. Após o download, os arquivos permanecem no cache local do Hugging Face.

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

Total original antes da auditoria:

**6.565 imagens**

## Baroque

A classe `Baroque` utiliza diretamente:

- `Baroque`

Total original antes da auditoria:

**4.240 imagens**

## Impressionism

A classe `Impressionism` utiliza diretamente:

- `Impressionism`

Total original antes da auditoria:

**13.060 imagens**

## Cubism

A classe `Cubism` agrega:

- `Cubism`
- `Analytical_Cubism`
- `Synthetic_Cubism`

Total original antes da auditoria:

**2.561 imagens**

O conjunto candidato às quatro macroclasses possui, portanto:

**26.426 imagens**

As regras de agrupamento são centralizadas em:

```text
src/data/labels.py
```

Não devem ser redefinidas manualmente em diferentes notebooks.

---

# Estado atual do projeto

Atualmente estão concluídas:

- **Etapa 1 — Análise Exploratória dos Dados**
- **Etapa 2 — Auditoria de Integridade do Dataset**

A próxima fase é:

**Etapa 3 — Curadoria e construção reproduzível do subconjunto final balanceado**

---

# Etapa 1 — Análise Exploratória

A análise exploratória investigou:

- quantidade de imagens por estilo;
- distribuição de artistas;
- distribuição dos gêneros de pintura;
- formação das macroclasses;
- presença de `Unknown Artist`;
- presença de `Unknown Genre`;
- possíveis associações entre gênero e movimento artístico;
- potenciais fontes de *shortcut learning*.

O notebook principal da etapa é:

```text
notebooks/01_analise_exploratoria.ipynb
```

Entre os principais achados estão:

- forte associação entre Renaissance e pinturas religiosas;
- forte associação entre Impressionism e `landscape`;
- presença relevante de `still_life` em Cubism;
- grande quantidade de registros `Unknown Artist`;
- diferenças importantes na distribuição de gêneros entre as classes.

Essas diferenças serão consideradas posteriormente na avaliação dos modelos, pois podem permitir que o classificador utilize características correlacionadas com as classes sem necessariamente aprender características estilísticas.

---

# Etapa 2 — Auditoria de Integridade

O notebook principal da auditoria é:

```text
notebooks/02_auditoria_dataset.ipynb
```

Foram auditadas todas as **26.426 imagens candidatas**.

## Integridade das imagens

Resultados:

- 26.426 imagens analisadas;
- 0 erros de abertura ou decodificação;
- 0 imagens com dimensões inválidas;
- 0 imagens com largura ou altura inferior a 256 pixels.

Todas as imagens candidatas são, portanto, tecnicamente compatíveis com a estratégia de geração de patches de `256×256`.

---

## Dimensões e resolução

A auditoria também analisou:

- largura;
- altura;
- quantidade total de pixels;
- megapixels;
- proporção largura/altura;
- orientação geométrica.

Foi observada forte recorrência de dimensões envolvendo aproximadamente **1.382 pixels**, indicando provável padronização ou limitação anterior das reproduções digitais.

A origem exata desse processamento não é inferida apenas a partir dos dados.

As quatro macroclasses apresentaram resoluções médias relativamente próximas, não havendo evidência de uma diferença dimensional extrema entre as classes.

---

## Orientação das imagens

Foi encontrada uma associação relevante entre orientação geométrica e macroclasse.

O Impressionismo apresenta predominância de imagens horizontais, enquanto Renaissance, Baroque e Cubism apresentam predominância de imagens verticais.

Esse resultado é importante porque a orientação poderá funcionar como uma variável correlacionada com a classe e deverá ser considerada posteriormente durante a investigação de *shortcut learning*.

---

# Auditoria de duplicatas

A redundância do dataset foi investigada em duas etapas.

## Duplicatas exatas

Foi calculado um **SHA-256** sobre:

- dimensões da imagem;
- pixels RGB decodificados.

Os hashes repetidos foram posteriormente confirmados por comparação pixel a pixel.

Resultado:

- 17 grupos de duplicatas exatas;
- 34 imagens envolvidas;
- nenhum conflito entre macroclasses.

---

## Variantes digitais da mesma obra

Também foi implementado um **hash perceptual baseado em DCT (pHash)**.

Foram investigados candidatos com distância de Hamming menor ou igual a 4.

O pHash foi utilizado apenas como mecanismo de triagem. Nenhuma imagem foi descartada automaticamente com base no hash perceptual.

Foram encontrados:

- distância 0: 20 pares;
- distância 2: 49 pares;
- distância 4: 67 pares.

Total:

**136 pares candidatos**

Todos os candidatos foram inspecionados visualmente.

Resultados:

- 135 pares confirmados como variantes da mesma obra;
- 1 falso positivo.

Portanto, o limiar utilizado apresentou alta precisão nesta auditoria específica, mas não é tratado como uma regra universal.

---

# Resultado final da auditoria

As duplicatas exatas e as variantes digitais confirmadas foram consolidadas em grupos únicos.

Resultado:

- 147 grupos de obras duplicadas;
- 297 registros envolvidos;
- 150 registros excedentes;
- 0 grupos envolvendo macroclasses diferentes.

Após a remoção dos registros redundantes:

**26.276 obras permanecem aprovadas pela auditoria.**

A política utilizada para escolher qual versão preservar foi:

1. maior quantidade total de pixels;
2. em caso de empate, maior largura;
3. persistindo o empate, menor `Indice_Dataset`.

---

# Arquivos de metadados

Os resultados da auditoria são registrados em:

```text
data/metadata/
```

## `auditoria_integridade.csv`

Contém o resultado completo da auditoria para todas as 26.426 obras candidatas.

Entre as informações registradas estão:

- índice original no WikiArt;
- estilo original;
- macroclasse;
- largura;
- altura;
- status de decodificação;
- validade dimensional;
- resolução;
- orientação;
- SHA-256;
- pHash;
- grupo de duplicação;
- status final da auditoria;
- motivo de descarte, quando aplicável.

---

## `candidatos_quase_duplicatas.csv`

Contém os pares encontrados pelo pHash.

Inclui:

- índices das duas imagens;
- estilos;
- macroclasses;
- distância pHash;
- classificação manual;
- observação da revisão.

---

## `grupos_duplicatas_confirmadas.csv`

Contém os grupos consolidados de registros que representam a mesma obra.

Também identifica qual registro foi preservado em cada grupo.

---

## `descartes_auditoria.csv`

Contém apenas os registros efetivamente removidos durante a auditoria.

Para cada descarte são armazenados:

- índice da imagem descartada;
- macroclasse;
- grupo de duplicação;
- índice da versão preservada;
- motivo do descarte;
- critério utilizado para selecionar a versão mantida.

---

# Estrutura do projeto

A estrutura utilizada atualmente é:

```text
tcc_wikiart/
│
├── README.md
├── requirements.txt
├── .gitignore
│
├── notebooks/
│   ├── 01_eda_wikiart.ipynb
│   └── 02_auditoria_dataset.ipynb
│
├── src/
│   └── data/
│       └── labels.py
│
├── data/
│   └── metadata/
│       ├── auditoria_integridade.csv
│       ├── candidatos_quase_duplicatas.csv
│       ├── grupos_duplicatas_confirmadas.csv
│       └── descartes_auditoria.csv
│
└── docs/
    └── project_log.md
```

Novas pastas e módulos serão adicionados conforme as próximas etapas forem implementadas.

---

# Onde modificar cada parte do projeto

Esta seção é especialmente importante para os integrantes do grupo.

## Alterações nas classes

Qualquer mudança relacionada às classes, subestilos ou agrupamentos deve ser feita em:

```text
src/data/labels.py
```

Esse arquivo é a fonte central das definições de classe.

Não criar mapeamentos diferentes diretamente nos notebooks.

---

## Análise exploratória

Alterações ou novas análises relacionadas ao dataset original devem ser feitas em:

```text
notebooks/01_analise_exploratoria.ipynb
```

---

## Auditoria de integridade

Alterações relacionadas a:

- imagens corrompidas;
- dimensões;
- resolução;
- hashes;
- duplicatas;
- critérios de descarte;

devem ser realizadas em:

```text
notebooks/02_auditoria_dataset.ipynb
```

---

## Histórico de decisões

As principais decisões metodológicas devem ser registradas em:

```text
docs/project_log.md
```

Esse arquivo funciona como um histórico cronológico das decisões tomadas durante o desenvolvimento.

---

# Próxima etapa

A próxima etapa é a construção reproduzível do subconjunto final.

Ela deverá utilizar **somente imagens aprovadas pela auditoria**.

Entre as próximas tarefas estão:

- definir o número definitivo de imagens por macroclasse;
- produzir um conjunto balanceado;
- preservar proporcionalmente os subestilos de Renaissance e Cubism;
- definir uma semente aleatória fixa;
- registrar exatamente quais índices foram selecionados;
- verificar novamente a distribuição de gêneros após a seleção;
- gerar o manifesto definitivo do dataset experimental.

Posteriormente serão realizadas:

- divisão entre treino, validação e teste;
- geração de patches;
- treinamento de uma CNN baseada em ResNet;
- treinamento de Vision Transformer;
- avaliação quantitativa;
- análise de *shortcut learning*;
- Grad-CAM;
- agregação das previsões dos patches no nível da pintura.

---

# Regra importante sobre divisão dos dados

A divisão entre:

- treino;
- validação;
- teste;

deverá ocorrer **no nível da pintura original antes da geração dos patches**.

Isso impede que patches provenientes da mesma obra sejam distribuídos entre treino e teste, o que provocaria vazamento de dados.

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

As dependências de treinamento serão adicionadas conforme a implementação dos modelos.

---

# Executando os notebooks

Os notebooks devem ser executados na ordem:

```text
01_eda_wikiart.ipynb
        ↓
02_auditoria_dataset.ipynb
        ↓
próximas etapas
```

Abra o notebook utilizando Jupyter Notebook, JupyterLab ou Visual Studio Code e execute as células na ordem apresentada.

Na primeira execução de:

```python
dataset = load_dataset("huggan/wikiart")
```

o dataset será baixado automaticamente.

---

# Reprodutibilidade

O dataset original não é armazenado no GitHub devido ao seu tamanho.

O repositório deve armazenar apenas os artefatos necessários para reproduzir os experimentos, como:

- índices das imagens selecionadas;
- decisões de auditoria;
- divisão entre treino, validação e teste;
- sementes aleatórias;
- configurações de treinamento;
- métricas;
- versões e parâmetros utilizados.

Sempre que possível, as decisões devem ser reproduzíveis a partir do dataset original e dos arquivos versionados neste repositório.

---

# Boas práticas para contribuição

Antes de iniciar alterações:

```bash
git pull --rebase origin main
```

Evitar modificar simultaneamente o mesmo notebook quando outro integrante estiver trabalhando nele.

Antes de realizar um commit:

1. executar as células alteradas;
2. verificar se o notebook continua funcionando;
3. remover outputs muito grandes;
4. confirmar que arquivos de dados pesados não serão enviados;
5. atualizar `project_log.md` quando houver uma decisão metodológica relevante.

Os notebooks devem preferencialmente ser versionados sem grandes outputs de imagens ou tabelas.

Arquivos como:

- dataset original;
- imagens;
- cache do Hugging Face;
- checkpoints de modelos;

não devem ser adicionados ao GitHub.

---

# Fluxo de trabalho do grupo

Antes de começar uma nova tarefa, recomenda-se verificar:

1. o `README.md`, para entender a estrutura atual;
2. o `docs/project_log.md`, para verificar decisões já tomadas;
3. o notebook correspondente à etapa;
4. os arquivos existentes em `src/`, para evitar duplicar regras ou funções.

Se uma lógica for reutilizada em diferentes etapas, ela deve preferencialmente ser movida para `src/`.

Se uma análise for específica de uma etapa experimental, ela pode permanecer no notebook correspondente.

---

# Histórico do projeto

O histórico cronológico das principais decisões metodológicas e resultados pode ser consultado em:

```text
docs/project_log.md
```

Esse documento deve ser atualizado sempre que uma decisão relevante alterar o desenho experimental do projeto.