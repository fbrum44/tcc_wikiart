# Project Log

Registro cronológico das principais decisões, resultados e mudanças metodológicas do projeto.

---

## 2026-10 — Definição do escopo experimental

- Definidas quatro macroclasses finais:
  - Renaissance
  - Baroque
  - Impressionism
  - Cubism
- Renaissance passou a agregar:
  - Early Renaissance
  - High Renaissance
  - Northern Renaissance
  - Mannerism / Late Renaissance
- Cubism passou a agregar:
  - Cubism
  - Analytical Cubism
  - Synthetic Cubism
- Decidido utilizar Transfer Learning.
- Definida geração de patches de 256×256 pixels.
- Definida avaliação comparativa entre CNN da família ResNet e Vision Transformer.
- Grad-CAM definido como principal técnica de interpretabilidade.
- Split deverá ocorrer no nível da pintura antes da geração dos patches.

---

## 2026-10 — Etapa 1: análise exploratória do WikiArt

- Dataset WikiArt carregado com 81.444 imagens.
- Identificadas 26.426 obras pertencentes aos estilos selecionados.
- Contagens antes da auditoria:
  - Renaissance: 6.565
  - Baroque: 4.240
  - Impressionism: 13.060
  - Cubism: 2.561
- Identificada forte associação entre gênero artístico e macroclasse.
- Renaissance apresentou forte concentração de obras religiosas.
- Impressionism apresentou forte concentração de landscapes.
- Cubism apresentou concentração relevante de still lifes.
- `Unknown Artist` foi considerado inadequado como critério principal de controle de autoria.
- `Unknown Genre` foi mantido e documentado como limitação dos metadados.
- Decidido não equalizar artificialmente a distribuição de gêneros entre as classes.
- Definida futura análise controlada por gênero, com retratos como principal candidato.

---

## 2026-11-02 — Etapa 2: auditoria de integridade

- Auditadas 26.426 imagens candidatas.
- Nenhum erro de abertura ou decodificação encontrado.
- Nenhuma dimensão inválida encontrada.
- Nenhuma imagem possui largura ou altura inferior a 256 pixels.
- Identificada forte recorrência de dimensões envolvendo aproximadamente 1.382 pixels.
- As quatro classes apresentaram resoluções digitais semelhantes.
- Identificada associação entre orientação e macroclasse:
  - Impressionism predominantemente horizontal;
  - Renaissance, Baroque e Cubism predominantemente verticais.
- Orientação registrada como possível fonte adicional de shortcut learning.

### Duplicatas exatas

- SHA-256 calculado sobre dimensões e pixels RGB decodificados.
- Encontrados 17 grupos de duplicatas exatas.
- 34 registros envolvidos.
- Confirmação realizada por comparação pixel a pixel.
- Nenhum grupo apresentou conflito de macroclasse.

### Quase-duplicatas

- Implementado pHash de 64 bits baseado em DCT.
- Utilizado limiar de distância de Hamming ≤ 4 apenas para triagem.
- 136 pares candidatos não exatos encontrados:
  - distância 0: 20;
  - distância 2: 49;
  - distância 4: 67.
- Todos os candidatos foram inspecionados manualmente.
- 135 pares foram confirmados como variantes da mesma obra.
- 1 par de distância 4 foi identificado como falso positivo.
- Decidido não realizar remoção automática baseada apenas em pHash.

### Consolidação

- Duplicatas exatas e variantes digitais foram consolidadas em componentes conexos.
- Resultado:
  - 147 grupos únicos;
  - 297 registros envolvidos;
  - 150 registros excedentes.
- Nenhum grupo envolveu mais de uma macroclasse.
- Política de preservação:
  1. maior quantidade total de pixels;
  2. maior largura em caso de empate;
  3. menor `Indice_Dataset` em novo empate.
- Após a auditoria, restam 26.276 registros aprovados.

---

## Próxima etapa

Etapa 3 — Curadoria e construção reproduzível do subconjunto final balanceado.