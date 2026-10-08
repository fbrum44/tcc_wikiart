# Classificação de Movimentos Artísticos com Visão Computacional

Projeto de Trabalho de Conclusão de Curso voltado à classificação automática
de movimentos artísticos a partir de pinturas digitalizadas utilizando técnicas
de Visão Computacional e Deep Learning.

O projeto também pretende utilizar técnicas de Inteligência Artificial
Explicável (XAI), como Grad-CAM, para investigar quais regiões das imagens
influenciam as decisões do modelo.

## Dataset

O projeto utiliza o dataset público WikiArt disponibilizado no Hugging Face:

`huggan/wikiart`

O conjunto contém 81.444 obras e possui os seguintes atributos:

- `image`: imagem da obra;
- `artist`: artista associado à obra;
- `genre`: gênero da pintura;
- `style`: movimento ou estilo artístico.

O dataset possui 27 classes de estilos artísticos.

As imagens não são armazenadas neste repositório. O dataset é baixado
diretamente do Hugging Face utilizando a biblioteca `datasets`.

```python
from datasets import load_dataset

dataset = load_dataset("huggan/wikiart")

--------------------------------------------------------------------------------
Na primeira execução, o download pode demorar devido ao tamanho do dataset.
Após o download, os arquivos são mantidos no cache local do Hugging Face.

Etapa atual
O projeto encontra-se na fase de análise exploratória e preparação dos dados.
Até o momento foram realizadas análises sobre:
- quantidade de imagens por movimento artístico;
- distribuição de artistas por movimento;
- distribuição dos gêneros de pintura dentro dos movimentos;
- possíveis vieses relacionados à associação entre gênero e movimento artístico.

Atualmente estão sendo investigados os seguintes movimentos:
- Renascimento;
- Barroco;
- Impressionismo;
- Cubismo.
O Renascimento ainda está sendo analisado por meio das quatro classes
originais presentes no WikiArt:
- Early_Renaissance;
- High_Renaissance;
- Northern_Renaissance;
- Mannerism_Late_Renaissance.
Essas classes ainda não foram definitivamente agrupadas em uma única classe.

----------------------------------------------------------------------------------
Configuração do ambiente
1. Clonar o repositório
git clone https://github.com/fbrum44/tcc_wikiart.git

Entre na pasta:
cd Wikiart

2. Criar um ambiente virtual
No Windows:
python -m venv .venv

Ativar o ambiente:
.venv\Scripts\activate

3. Instalar as dependências
pip install -r requirements.txt

4. Executar o notebook
Abra o arquivo .ipynb utilizando Jupyter Notebook ou Visual Studio Code
e execute as células na ordem apresentada.
Na primeira execução do comando:
dataset = load_dataset("huggan/wikiart")
o dataset será baixado automaticamente.

O dataset original não será armazenado no GitHub devido ao seu tamanho.
Após a definição das classes e da divisão dos dados, o projeto deverá armazenar
apenas informações necessárias para reproduzir os experimentos, como índices
das imagens selecionadas, divisão entre treino, validação e teste e sementes
aleatórias utilizadas.

Observações
Os notebooks do repositório devem preferencialmente ser enviados sem grandes
outputs de imagens ou tabelas, evitando aumentar desnecessariamente o tamanho
dos arquivos .ipynb.
