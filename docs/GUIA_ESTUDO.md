# Guia de estudo — fundamentos e implementação das Partes 0 e 1

Este material concentra os fundamentos e a implementação das Partes 0 e 1. Comece pela Parte 0 e acompanhe a Parte 1 em `experimentos_pa2.ipynb`. As referências de funções abaixo são relativas à raiz do repositório. Para as Partes 2–5, consulte os notebooks integrados e `docs/RESULTADOS_INTEGRADOS.md`.

## 1. O problema que estamos resolvendo

O detector responde: **onde há uma pessoa neste quadro?** O rastreador responde: **qual pessoa é esta ao longo dos quadros?**

Imagine uma pessoa que aparece em três imagens consecutivas. O detector pode fornecer três caixas corretas. Ainda precisamos associá-las à mesma identidade. Se o rastreador atribuir IDs 1, 2 e 3, detectou a pessoa, mas perdeu a continuidade e pode contá-la três vezes.

Uma **track** é o registro mantido pelo rastreador para uma identidade. No nosso baseline, contém a última caixa e o quadro da última observação. Não é uma rede neural.

O **ground truth**, abreviado GT, é a resposta anotada: posição e identidade verdadeira de cada objeto. Serve para supervisionar ou avaliar. Na inferência da Parte 1, nunca é fornecido ao rastreador.

## 2. Como ler os dados

### Caixa em xywh

Usamos `x,y,w,h`: posição do canto superior esquerdo, largura e altura. Não são quatro coordenadas de cantos.

```text
x=30, y=40, w=12, h=16
canto superior esquerdo: (30, 40)
canto inferior direito: (42, 56)
```

As posições internas são 0-based. Quadros e IDs começam em 1. O formato MOT usa posições 1-based: `mot_data.py` subtrai 1 de x/y ao ler e soma 1 ao exportar. Não altera largura ou altura.

### Os três formatos principais

| Dado | Colunas internas | Significado |
| --- | --- | --- |
| GT sintético | `frame,id,x,y,w,h,visibility` | Objeto e sua visibilidade em um quadro |
| Detecção | `frame,x,y,w,h,score` | Caixa encontrada, sem identidade |
| Predição de tracking | `frame,id,x,y,w,h` | Caixa à qual o rastreador atribuiu um ID |

`frame` indica o instante discreto do vídeo. `id` identifica o objeto, não o vídeo. `score` é a confiança do detector; não é a confiança de que duas caixas pertencem à mesma pessoa.

O número do ID previsto não precisa ser igual ao ID do GT. Se a pessoa 2 do GT recebe o ID 7 durante toda a sequência, pode estar perfeitamente rastreada.

## 3. Parte 0 — por que criar um ambiente sintético?

Em vídeos reais é difícil descobrir se um resultado estranho vem do detector, do rastreador ou da avaliação. No sintético conhecemos a resposta e controlamos a dificuldade.

O PDF pede quatro componentes: gerador, detector simulado, métricas com testes manuais e baseline no piso fácil. Também pede variar a dificuldade e demonstrar oclusão real.

### 3.1 Gerador

Local: `synthetic.py`, função `generate`.

```python
def generate(n_objects=5, speed=0.3, occlusion=0, n_frames=48,
             noise=2.0, contrast=1.0, seed=0):
```

Produz uma `Scene` com `frames` e `gt`. As imagens têm 128×128 pixels. O gerador aceita 5–15 objetos e 30–60 quadros. As elipses têm tamanhos e cores variados e refletem nas bordas.

| Parâmetro | O que controla | Observação |
| --- | --- | --- |
| `n_objects` | Quantidade de elipses | Pode aumentar sobreposições |
| `speed` | Velocidade em pixels/quadro | Aumenta o deslocamento entre observações |
| `occlusion` | Duração programada de uma oclusão específica | Não é a quantidade total de oclusões da cena |
| `n_frames` | Duração do vídeo | Mais duração não implica maior dificuldade em cada quadro |
| `noise` | Ruído nos pixels | Altera a imagem |
| `contrast` | Diferença visual entre objetos e fundo | Altera a imagem |
| `seed` | Inicialização do gerador aleatório | Permite repetir o experimento |

**Atenção:** o detector sintético gera caixas a partir do GT; não examina a imagem. Assim, ruído e contraste dos pixels não alteram diretamente suas detecções. A corrupção das caixas é controlada separadamente.

### 3.2 Oclusão real e visibilidade

Uma elipse maior é desenhada à frente de outra. Os pixels do objeto da frente substituem os pixels do objeto de trás, como no exemplo de um eclipse.

Trecho de `synthetic.py:generate`:

```python
for i, mask in enumerate(masks):
    ownership[mask] = i
    canvas[mask] = 25 + contrast*(colors[i]-25)
```

`mask` indica os pixels de uma elipse. `ownership` registra qual objeto ficou visível em cada pixel após todos serem desenhados.

`visibility` é a fração dos pixels da elipse ainda pertencente a ela no resultado final: 1 significa totalmente visível, 0 significa totalmente escondida. Com `occlusion=10`, o objeto 1 desaparece completamente por dez quadros e depois volta.

O GT registra inclusive o objeto oculto. Porém, **a avaliação sintética usa somente observações com visibilidade maior que zero**. Essa regra é diferente da avaliação MOT usada na Parte 1.

Remover uma caixa do detector não comprova oclusão visual: a pessoa poderia continuar visível na imagem. Nosso exemplo comprova desaparecimento nos pixels.

### 3.3 Detector simulado

Local: `synthetic.py`, função `corrupt_detections`.

```python
def corrupt_detections(gt, drop=0.0, jitter=0.0,
                       false_positives=0.0, seed=0):
```

- `drop`: probabilidade de descartar uma caixa visível, simulando uma detecção perdida.
- `jitter`: ruído gaussiano, em pixels, aplicado aos componentes da caixa.
- `false_positives`: média da distribuição Poisson que decide quantas caixas falsas inserir por quadro.

Descartamos **caixas**, não quadros inteiros. Um quadro com cinco objetos pode ter apenas quatro detecções. Um falso positivo é uma detecção sem objeto correspondente; não deve ser confundido com um novo objeto real.

O simulador usa o GT para criar um experimento controlado. Sua saída não contém IDs. O rastreador recebe apenas as caixas.

### 3.4 Execução e resultados

Local: `run_synthetic.py`, funções `run` e `run_experiments`.

`run` simula detecções, atualiza o baseline em cada quadro e chama a avaliação. `run_experiments` organiza os experimentos, gera figuras e salva JSON. O notebook chama:

```python
synthetic_results = run_experiments("outputs/parte0")
```

Resultados de referência:

- Piso fácil: IDF1=1, zero switches e 5 IDs previstos para 5 objetos.
- A 5 pixels/quadro: IDF1 médio de 0,4731.
- A 9 pixels/quadro: IDF1 médio de 0,0403.
- De 5 a 15 objetos, com o movimento lento padrão: IDF1 permanece 1. Quantidade isolada não quebrou o baseline neste experimento.

Cada eixo varia um parâmetro por vez, com três seeds. As barras são desvio padrão amostral entre seeds, não intervalo de confiança.

A curva de oclusão não é monotônica. Quando aumenta a duração, removemos mais observações totalmente ocultas da avaliação sintética, mudando a ponderação do IDF1. Nos casos de 4, 10 e 20 quadros, aparece um switch e um ID extra. A pequena subida de IDF1 não demonstra recuperação da identidade.

## 4. Associação: IoU e Hungarian

### 4.1 IoU

Local: `geometry.py`, função `iou_matrix`.

```text
IoU = área da interseção / área da união
```

Caixas iguais têm IoU=1. Sem sobreposição, IoU=0. Para duas caixas 10×10, deslocadas horizontalmente por 5 pixels, a interseção é 50 e a união é 150: IoU=1/3.

IoU mede sobreposição espacial, não identidade. Pessoas diferentes podem ter caixas sobrepostas; a mesma pessoa pode se mover tanto que sua caixa não se sobreponha à anterior.

### 4.2 Associação um para um

Local: `geometry.py`, função `match_iou`.

Construímos uma matriz: linhas para tracks existentes, colunas para novas detecções. Só aceitamos pares acima do limiar. O solver Hungarian, do SciPy, encontra uma atribuição global um para um.

A lógica de associação é própria: **maximizar primeiro a quantidade de pares válidos e depois a soma dos IoUs**. Isso evita que várias tracks recebam a mesma detecção. Usar o solver permitido do SciPy não equivale a usar um rastreador pronto.

## 5. O baseline passo a passo

Local: `baseline.py`, classe `IoUTracker`, método `update`.

Configuração da Parte 1: `threshold=0.3`, `max_age=2`.

Para cada quadro:

1. Verifica que os quadros são crescentes.
2. Remove tracks antigas demais.
3. Compara a última caixa de cada track com as detecções atuais.
4. Mantém o ID dos pares associados.
5. Cria um novo ID para cada detecção sem par.
6. Atualiza a caixa e o instante da última observação.

Trecho da expiração em `baseline.py:IoUTracker.update`:

```python
self.tracks = {
    i: t for i, t in self.tracks.items()
    if frame - t[1] <= self.max_age + 1
}
```

Com `max_age=2`, uma track vista no quadro 1 pode ficar ausente nos quadros 2 e 3 e ainda ser associada no 4. Se voltar apenas no 5, a track antiga já expirou. Mesmo dentro da tolerância, precisa passar no limiar de IoU.

Tracks ausentes ficam temporariamente na memória, mas não emitem caixas. O baseline não prevê movimento, não estima velocidade e não usa características de aparência.

Local: `parte1.py`, função `run_tracker`. Processa todos os quadros, inclusive quando a entrada é vazia. Um quadro sem caixas não faz o tempo parar.

## 6. Métricas: como distinguir os erros

Local: `metrics.py`, função `evaluate`. Recebe GT e predições em seis colunas e retorna IDF1, switches, fragmentações e contagens.

### 6.1 IDF1

O código acumula compatibilidade espacial entre IDs ao longo da sequência e encontra uma correspondência global um para um entre identidades verdadeiras e previstas. O limiar espacial padrão é IoU=0,5.

```text
IDF1 = 2 × IDTP / (2 × IDTP + IDFP + IDFN)
```

`IDTP` conta observações compatíveis sob a correspondência global. `IDFP` e `IDFN` representam observações previstas e verdadeiras que não foram creditadas nessa correspondência.

Exemplo: uma pessoa observada em dez quadros recebe o ID 1 nos primeiros cinco e o ID 2 nos últimos cinco. A correspondência global só pode escolher um desses IDs para a pessoa. Portanto, caixas perfeitas não garantem IDF1 perfeito.

### 6.2 Switch

Um GT que estava associado a um ID passa a outro ID. No nosso protocolo, a comparação considera a última correspondência válida, inclusive depois de uma lacuna.

Não é a mesma coisa que contar IDs novos. Um ID novo pode ser um nascimento correto; um switch envolve a mudança de associação de uma identidade verdadeira.

### 6.3 Fragmentação

Uma identidade do GT estava associada, deixa de ser associada em uma observação e posteriormente volta a ser. A fragmentação conta a interrupção e retomada.

Uma pessoa passar diretamente do ID 1 para o ID 2, sem quadro perdido, produz switch, mas não necessariamente fragmentação. Uma pessoa perder uma detecção e voltar com o mesmo ID pode produzir fragmentação sem switch.

Uma ausência de linha do GT não conta como observação perdida. Isso é especialmente relevante ao excluir objetos totalmente ocultos no sintético.

### 6.4 Testes manuais

Locais: tabela da Parte 0 no notebook e `tests/test_core.py`.

| Caso: duas pessoas, quatro quadros, caixas perfeitas | IDF1 | Switches | Fragmentações |
| --- | ---: | ---: | ---: |
| Previsão igual ao GT | 1,00 | 0 | 0 |
| Dois IDs trocados a partir do quadro 3 | 0,50 | 2 | 0 |
| Somente uma identidade dividida a partir do quadro 3 | 0,75 | 1 | 0 |

O primeiro caso comprova o comportamento perfeito. Os dois seguintes verificam que situações diferentes têm impactos distintos, mesmo sem erros nas caixas.

### 6.5 Contagem

Contamos IDs únicos previstos e IDs reais. O erro de contagem é o valor absoluto da diferença. É possível acertar a contagem e trocar todas as identidades, portanto contagem não substitui IDF1.

## 7. Parte 1 — aplicação ao MOT17

### 7.1 Duas fontes, mesmo baseline

O PDF exige detecções públicas e um detector torchvision pré-treinado.

- Fonte pública: FRCNN do MOT17, escolhida e congelada para as próximas partes.
- Torchvision: Faster R-CNN ResNet-50 FPN, pesos COCO_V1, classe `person`, sem fine-tuning.

Local: `detector.py`, função `build_detector`. As imagens são necessárias ao detector; a fonte pública já fornece as caixas e pode ser avaliada sem imagens.

Treino/desenvolvimento: 04/05/10/11. Validação: 02/09/13. As variantes DPM/FRCNN/SDP de um vídeo físico não cruzam splits. Nesta parte, esses nomes de splits não implicam treinamento do baseline.

Local: `configs/parte1.json`. Configuração central:

| Parâmetro | Valor | Uso |
| --- | ---: | --- |
| `score_threshold` | 0,5 | Aceitar detecções para avaliação e tracking |
| `association_iou` | 0,3 | Associar detecção a track |
| `max_age` | 2 | Tolerar quadros intermediários sem observação |
| `evaluation_iou` | 0,5 | Compatibilidade espacial na métrica de tracking |
| `ignore_iou` | 0,5 | Tratamento das anotações ignoradas |
| `cache_score_threshold` | 0,05 | Preservar mais caixas no cache, antes do corte final |

Não confundir o limiar de associação com o de avaliação. O score também não é IoU: um vem do detector, o outro da geometria entre caixas.

### 7.2 NMS próprio

Local: `geometry.py`, função `nms`; integração em `detector.py`, função `own_nms_context`.

NMS remove caixas redundantes: mantém a de maior score e suprime caixas muito sobrepostas. É uma operação sobre detecções no quadro, diferente de associar identidades no tempo.

O contexto substitui o NMS do detector por nossa implementação, inclusive nas etapas internas, e restaura as funções ao sair. Isso atende à proibição de `torchvision.ops.nms`.

### 7.3 Cache

Locais: `detector.py`, funções `extract_detections` e `load_cached_detections`.

Guardamos detecções por quadro, incluindo quadros vazios, e um manifesto com hashes de conteúdo, código e configuração. Um hash é uma identificação do conteúdo que permite detectar mudanças. Isso evita executar a rede a cada avaliação e permite retomar inferência interrompida.

Um cache incompleto não sustenta a comparação final. Os três vídeos inteiros de validação somam 1.875 quadros.

### 7.4 GT só na avaliação

Local: `parte1.py`, função `evaluate_sequence`.

```text
detecções → corte de score → baseline → previsões
                                      ↓
                    GT → filtro de avaliação → métricas
```

Local: `mot_data.py`, funções `valid_gt` e `evaluation_mask`. Avaliamos pedestres válidos, sem corte de visibilidade. Predições remanescentes associadas a anotações ignoradas são desconsideradas depois de proteger os matches com pedestres válidos.

Filtrar entradas do rastreador usando GT vazaria respostas. Nosso protocolo é `pa2-pedestrians-v1`; não afirmamos equivalência integral ao avaliador oficial MOTChallenge.

## 8. AP50 e mAP: avaliação espacial

Local: `detection_metrics.py`, função `evaluate_detection`.

Precisão mede a proporção de detecções consideradas corretas entre as previstas. Recall mede a proporção de objetos verdadeiros encontrados. Ao ordenar as detecções por score, obtemos uma curva precisão–recall.

AP resume essa curva. Implementamos interpolação em 101 pontos de recall, com associação espacial independente por quadro.

- AP50 usa IoU de pelo menos 0,5.
- mAP 0,50:0,95 é a média dos APs em dez limiares: 0,50, 0,55, …, 0,95. Existe apenas uma classe alvo.

Nosso cálculo usa somente detecções acima do corte 0,5 e um tratamento próprio dos ignorados. Assim, não é uma reprodução integral do COCO. AP não verifica continuidade de identidade.

## 9. Como interpretar nossos resultados

Valores da execução local com caches CPU registrada no notebook unificado:

| Sequência | Fonte | AP50 | mAP | IDF1 | IDs previstos / GT |
| --- | --- | ---: | ---: | ---: | ---: |
| MOT17-02 | Pública | 0,3363 | 0,2659 | 0,3632 | 135 / 62 |
| MOT17-02 | Torchvision | 0,4166 | 0,2338 | 0,3078 | 707 / 62 |
| MOT17-09 | Pública | 0,5544 | 0,4638 | 0,5515 | 50 / 26 |
| MOT17-09 | Torchvision | 0,7159 | 0,4320 | 0,4512 | 326 / 26 |
| MOT17-13 | Pública | 0,5607 | 0,3853 | 0,4640 | 518 / 110 |
| MOT17-13 | Torchvision | 0,5432 | 0,2628 | 0,3307 | 1248 / 110 |

Pequenas diferenças numéricas podem aparecer na execução CUDA do notebook individual. Identifique a execução ao apresentar os números.

Na MOT17-09, o AP50 melhora com torchvision, mas o IDF1 piora e há muito mais IDs. Essa é a demonstração principal: qualidade espacial no limiar 0,5 não garante identidade consistente com esse rastreador.

O mAP cai nas três sequências; portanto, não diga que torchvision melhorou toda a avaliação de detecção. A fonte pública tem maior IDF1 nas três sequências sob a configuração avaliada; isso não estabelece superioridade universal do detector.

Local: `visualization.py`, função `plot_descolamento`. O gráfico mostra mAP e IDF1 no painel superior. O inferior mostra IDs previstos/IDs reais e switches/IDs reais em eixos distintos. As sequências são ordenadas por densidade GT.

Os resultados agregados não isolam a causa das falhas. Movimento, oclusão e oscilação das caixas são hipóteses a investigar em trechos concretos.

## 10. Mapa dos arquivos

| Arquivo | Responsabilidade |
| --- | --- |
| `synthetic.py` | Gerador e detector simulado da Parte 0 |
| `run_synthetic.py` | Experimentos sintéticos e figuras |
| `geometry.py` | IoU, matching e NMS próprios |
| `baseline.py` | Estado e atualização do baseline |
| `metrics.py` | IDF1, switches, fragmentações e contagem |
| `mot_data.py` | Leitura, formatos, GT válido e ignorados |
| `detection_metrics.py` | AP50 e mAP |
| `detector.py` | Inferência torchvision, NMS, cache e benchmark |
| `parte1.py` | Pipeline de tracking e avaliação no MOT17 |
| `visualization.py` | Gráficos e exemplos de identidades |
| `experimentos_pa2.ipynb` | Experimentos apresentados com saídas salvas |

## 11. Perguntas para revisar sem consultar o texto

1. Por que caixas corretas podem coexistir com IDF1 baixo?
2. Que informação existe no GT e não deve entrar no rastreador?
3. Como se distingue oclusão real de uma detecção perdida?
4. Por que `noise` da imagem não piora diretamente nosso detector sintético?
5. Quais são as três corrupções do detector simulado?
6. O que IoU mede e o que ela não consegue garantir?
7. Por que a associação precisa ser um para um?
8. Uma track vista no quadro 1 pode retornar com o mesmo ID no 4? E no 5?
9. Há caixa prevista durante uma ausência no nosso baseline?
10. Qual a diferença entre switch e fragmentação?
11. Por que dividir uma identidade altera IDF1 mesmo com caixas perfeitas?
12. Por que acertar o número total de pessoas não garante bom tracking?
13. Qual a diferença entre score 0,5, IoU de associação 0,3 e IoU de avaliação 0,5?
14. NMS associa pessoas entre quadros?
15. Por que o mAP pode cair quando AP50 sobe?
16. Que conclusão a MOT17-09 permite? Que conclusão ela não permite?
17. Por que congelar detecções ao comparar baseline e modelo temporal?

## 12. Resumo para explicar em voz alta

“Primeiro construímos um ambiente sintético com respostas conhecidas para validar as métricas e o baseline. O rastreador associa novas detecções à última caixa de cada identidade por IoU e Hungarian, sem prever movimento. Depois aplicamos o mesmo baseline ao MOT17, comparando uma fonte pública e um detector pré-treinado. Medimos caixas com AP e identidades com IDF1, switches e fragmentações. Os resultados mostram que melhorar AP50 não garante melhor continuidade de identidade. Isso estabelece a referência para avaliar o modelo temporal com as mesmas detecções.”
