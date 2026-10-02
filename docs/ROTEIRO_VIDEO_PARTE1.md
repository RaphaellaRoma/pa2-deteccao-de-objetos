# Roteiro de vídeo — Parte 1: detecção e identidade ao longo do tempo

## Orientações para o NotebookLM

Produza um vídeo explicativo em português brasileiro, de aproximadamente 6–8 minutos, para um estudante que está aprendendo rastreamento de objetos. Use linguagem simples, exemplos visuais e o roteiro abaixo. Explique os termos antes de utilizá-los. Evite mostrar funções inteiras ou ler tabelas extensas.

Este documento descreve a implementação deste repositório. Não invente experimentos, treinamento ou resultados. A Parte 1 está executada; o modelo temporal pertence às partes seguintes. Os valores apresentados abaixo correspondem à execução local com caches CPU registrada no notebook unificado; pequenas diferenças em relação à execução CUDA do notebook individual são possíveis.

Fontes do projeto: `experimentos_pa2.ipynb`, `configs/parte1.json`, `baseline.py`, `geometry.py`, `mot_data.py`, `metrics.py`, `detection_metrics.py`, `detector.py`, `parte1.py` e `visualization.py`. O enunciado é `PA2.pdf`.

## Cena 1 — O problema: detectar não basta

**Narração:**

“Imagine um vídeo com várias pessoas andando. Um detector desenha uma caixa em volta de cada pessoa em cada imagem. Mas uma caixa não informa se aquela pessoa é a mesma que apareceu antes. O rastreador precisa ligar essas observações ao longo do tempo e manter uma identidade consistente.

Na Parte 1, construímos um baseline: um método simples que servirá de referência para o modelo temporal. Queremos medir se boas detecções também produzem boas identidades.”

**Visual:** três quadros com uma pessoa se deslocando. Primeiro mostrar somente caixas; depois repetir com o mesmo ID nos três quadros.

## Cena 2 — O que o enunciado pede e quais entradas usamos

**Narração:**

“O trabalho exige duas fontes de detecção: uma pública do MOT17 e um detector pré-treinado do torchvision. Escolhemos o FRCNN público como fonte congelada para as próximas partes. A comparação executada também mostrou maior IDF1 com essa fonte nos três vídeos de validação, sob o nosso baseline.

Na segunda fonte usamos Faster R-CNN com ResNet-50 e FPN, pesos COCO_V1 e somente a classe pessoa. Não treinamos nem ajustamos esse detector.

O split separa vídeos inteiros: 04, 05, 10 e 11 para desenvolvimento e treino posterior; 02, 09 e 13 para validação. Não dividimos quadros do mesmo vídeo entre os dois grupos. Nesta parte, não há treinamento.”

**Visual:** duas fontes de caixas chegando ao mesmo rastreador. Mostrar discretamente os grupos de sequências.

**Referências:** `configs/parte1.json`; `mot_data.py`, classe `Sequence`; `detector.py`, função `build_detector`.

## Cena 3 — Como o baseline associa as caixas

**Narração:**

“O baseline guarda a última caixa observada de cada identidade. Quando chega um novo quadro, compara essas caixas com as novas detecções usando IoU: a área de interseção dividida pela área da união. Caixas iguais têm IoU um; caixas sem sobreposição têm IoU zero.

Só permitimos associações com IoU de pelo menos 0,3. O algoritmo Hungarian resolve uma associação um para um: uma detecção não pode pertencer a duas tracks. Nossa regra maximiza primeiro a quantidade de pares válidos e depois a soma dos IoUs.

Se uma caixa não é associada, ela recebe um novo ID. Esse método não estima velocidade nem prevê onde a pessoa estará no próximo quadro.”

**Visual:** duas caixas antigas, duas novas e uma matriz de IoUs. Destacar os pares escolhidos e uma detecção sem par recebendo novo ID.

**Referências:** `baseline.py`, classe `IoUTracker`, método `update`; `geometry.py`, funções `iou_matrix` e `match_iou`.

## Cena 4 — O que acontece durante uma ausência

**Narração:**

“A configuração max_age igual a dois permite dois quadros intermediários sem observação. Se vimos uma pessoa no quadro 1, podemos não vê-la nos quadros 2 e 3 e ainda tentar recuperar seu ID no quadro 4. Mas a associação também precisa passar pelo limiar de IoU.

Durante a ausência, a track fica na memória, mas não emite uma caixa prevista. Se a ausência ultrapassa a tolerância, a track é removida. Uma detecção que reaparece pode então receber outro ID.

Isso explica uma fragilidade do baseline: a pessoa pode continuar existindo, mas o sistema passa a contá-la como uma nova identidade.”

**Visual:** linha do tempo com observação, dois quadros vazios e retorno; depois um segundo exemplo com ausência mais longa e nascimento de outro ID.

**Referências:** `baseline.py`, método `IoUTracker.update`; `parte1.py`, função `run_tracker`, que processa inclusive os quadros vazios.

## Cena 5 — Como medimos o resultado

**Narração:**

“Avaliamos duas propriedades diferentes. AP50 e mAP medem a qualidade espacial das detecções. AP50 usa IoU 0,5; o mAP combina dez limiares, de 0,5 até 0,95. Nossos cálculos usam as detecções com confiança de pelo menos 0,5 e um protocolo próprio, não o avaliador oficial COCO.

IDF1 mede a consistência das identidades considerando uma correspondência global ao longo da sequência. Switch conta quando uma pessoa passa a ser associada a outro ID. Fragmentação registra uma interrupção e posterior retomada das correspondências nas observações do GT.

Também contamos IDs únicos. Se existem 26 pessoas, mas geramos 326 IDs, houve grande excesso de identidades. Essa contagem sozinha não explica todos os erros, por isso a analisamos junto com IDF1 e switches.

O número do ID previsto não precisa ser igual ao número do ID verdadeiro. O importante é manter uma correspondência consistente.”

**Visual:** quadro dividido entre ‘caixas corretas’ e ‘identidades consistentes’. Mostrar uma pessoa com ID 7 durante todo o vídeo, embora seu ID no GT seja 2.

**Referências:** `detection_metrics.py`, função `evaluate_detection`; `metrics.py`, função `evaluate`.

## Cena 6 — Avaliação sem fornecer as respostas ao rastreador

**Narração:**

“O rastreador recebe somente detecções. O ground truth, ou resposta anotada, só entra depois para avaliar. As anotações ignoradas são tratadas nessa etapa, preservando primeiro as associações com pedestres válidos.

Na inferência do torchvision, usamos nosso próprio NMS para remover caixas redundantes, inclusive nas etapas internas do detector. O enunciado proíbe usar torchvision.ops.nms. Também não usamos rastreadores prontos nem bibliotecas prontas de métricas de tracking.”

**Visual:** fluxo: detecções → rastreador → previsões → avaliação. O GT deve entrar apenas na avaliação. Explicar NMS com duas caixas sobrepostas e uma sendo removida.

**Referências:** `parte1.py`, função `evaluate_sequence`; `mot_data.py`, função `evaluation_mask`; `geometry.py`, função `nms`; `detector.py`, função `own_nms_context`.

## Cena 7 — O resultado principal

**Narração:**

“A comparação cobre os três vídeos de validação completos: 600, 525 e 750 quadros, totalizando 1.875.

Na MOT17-09, trocar a fonte pública pelo torchvision aumenta AP50 de aproximadamente 0,5544 para 0,7159. Mas IDF1 cai de 0,5515 para 0,4512. A contagem passa de 50 para 326 IDs, para as mesmas 26 pessoas reais.

Portanto, melhorar a detecção no limiar IoU 0,5 não garantiu melhor continuidade de identidade. Isso não significa que toda a avaliação espacial melhorou: o mAP caiu nas três sequências. O público também obteve maior IDF1 nas três.

O gráfico do notebook mostra mAP e IDF1 no painel superior; no inferior, IDs previstos por identidade real e switches por identidade real. A ordem é definida pela densidade de pedestres. As métricas agregadas não demonstram sozinhas se a causa foi movimento, oclusão ou oscilação das caixas.”

**Visual:** destacar os números da MOT17-09 e depois mostrar `outputs/parte1/descolamento.png`. Identificar separadamente os dois eixos do painel inferior. Não sugerir que AP50 e IDF1 têm a mesma definição.

**Referências:** seção da Parte 1 em `experimentos_pa2.ipynb`; `visualization.py`, função `plot_descolamento`.

## Cena 8 — A conclusão e a próxima etapa

**Narração:**

“A Parte 1 estabelece uma referência reproduzível. O baseline funciona associando caixas próximas, mas não tem previsão de movimento nem memória aprendida. Por isso pode perder uma identidade quando a pessoa se desloca muito ou desaparece por tempo suficiente.

Nas próximas partes, o modelo temporal será comparado com esse baseline usando as mesmas detecções públicas congeladas e o mesmo protocolo. Assim podemos avaliar a contribuição do modelo temporal sem misturá-la com uma troca de detector.”

**Visual:** baseline e modelo temporal recebendo as mesmas caixas. Fechar com: ‘Detectar a pessoa é o primeiro passo; preservar sua identidade é o desafio temporal’.

## Cuidados ao produzir o vídeo

- Não apresentar as métricas como resultados oficiais do MOTChallenge ou COCO.
- Não afirmar que o baseline prevê movimento ou desenha caixas durante ausências.
- Distinguir IoU de associação, 0,3, de IoU de avaliação, 0,5.
- Não afirmar que o torchvision melhorou mAP: o ganho destacado é de AP50 na MOT17-09.
- Não apresentar excesso de IDs como uma contagem exata de switches.
- Não afirmar que os resultados isolam a causa de cada falha.
- Se usar resultados de outra execução, identificar a execução e atualizar os números de forma consistente.
