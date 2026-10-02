# Investigação: IDs extras do baseline com torchvision na MOT17-09

## Pergunta

Na Parte 1, torchvision tem AP50 maior que as detecções públicas na MOT17-09, mas IDF1 menor e muito mais identidades previstas. Testamos a hipótese simples de que as caixas do torchvision mudam tanto de um quadro para o seguinte que o IoU cai abaixo do limiar de associação 0,3.

## Método

Usamos os caches e previsões já existentes; não alteramos o notebook ou as métricas publicadas. A investigação é retrospectiva: em cada quadro, associamos detecções ao GT por IoU >= 0,5 para identificar detecções da mesma pessoa em quadros sucessivos. Essa associação com GT serve somente ao diagnóstico e nunca é fornecida ao rastreador.

Em seguida comparamos a caixa da mesma pessoa entre quadros, seu IoU, e os IDs que o baseline efetivamente previu. Também examinamos os intervalos entre observações GT em que o ID previsto mudou.

## Resultados na MOT17-09

| Medida | FRCNN público | Torchvision |
| --- | ---: | ---: |
| Detecções acima do score 0,5 | 2.944 | 6.083 |
| Detecções casadas a pedestres GT por quadro, IoU >= 0,5 | 2.932 | 3.911 |
| Pares da mesma pessoa em quadros consecutivos | 2.866 | 3.767 |
| IoU mediano da caixa entre pares consecutivos | 0,915 | 0,906 |
| Percentual de pares consecutivos com IoU < 0,3 | 0% | 0% |
| Trocas do ID previsto entre observações adjacentes de GT | 0 | 63 |

Depois fizemos uma reprodução instrumentada do `IoUTracker` quadro a quadro. Para cada mudança de ID detectada por comparação GT-predição, rastreamos a detecção exata que gerou a caixa prevista e verificamos se a track antiga ainda existia, seu IoU com a caixa atual e a associação escolhida pelo Hungarian.

| Diagnóstico para as 138 mudanças de ID | Casos |
| --- | ---: |
| Track antiga já expirada | 47 |
| IoU da track antiga com a detecção < 0,3 | 10 |
| IoU >= 0,3, mas Hungarian deu a detecção a outra track | 67 |
| IoU >= 0,3, mas a detecção ficou sem associação e gerou ID novo | 12 |
| Matching quadro a quadro ambíguo por caixas sobrepostas | 2 |

Assim, **79/138 casos** ocorreram mesmo com IoU >= 0,3: a decisão global não preservou a associação isolada com a track antiga. Isso é evidência direta do comportamento do Hungarian neste baseline, não prova de que uma característica específica do modelo torchvision sempre cause a competição.

Exemplo visual: o GT 1 passa da predição 126 no quadro 207 para a 125 no 208. A caixa correspondente à mesma pessoa ainda tinha IoU 0,615 com a track antiga. O Hungarian escolheu outra atribuição global. As caixas do detector podem se sobrepor entre si e competir por tracks.

![Quadros 207 e 208: GT verde e predições do baseline em vermelho tracejado](../outputs/parte1/examples/torchvision-switch-207-208.png)

As contagens de intervalos entre observações GT em que o ID mudou foram:

- FRCNN público: 36 mudanças no total; nenhuma ocorreu entre observações adjacentes. As demais foram após lacunas, muitas longas.
- Torchvision: 138 mudanças no total; 63 entre observações adjacentes e 75 após lacunas.

Essas contagens foram obtidas ao casar previsões ao GT quadro a quadro para inspeção. Não são exatamente a definição protocolar de `id_switches` usada pelo projeto; são uma decomposição diagnóstica por intervalo.

## Interpretação

A hipótese de que **a maioria das trocas vem de quedas simples do IoU da mesma caixa entre quadros consecutivos** não é sustentada nessa sequência: os pares consecutivos corretos mantêm sobreposição alta.

O diagnóstico sugere outras possibilidades:

1. O torchvision tem mais detecções acima do score: 6.083 contra 2.944. Mais caixas podem aumentar candidatos espúrios ou ambiguidades para o Hungarian.
2. Em 47 mudanças, a track antiga já havia expirado. Lacunas longas de detecção fazem o baseline esquecer a identidade.
3. Em 79 mudanças, a track antiga ainda tinha IoU >= 0,3 com a caixa-alvo, mas a associação global não preservou o par. Isso pode ocorrer quando outras tracks e detecções competem no mesmo quadro.

As categorias explicam mecanicamente os eventos observados, mas não isolam por que o torchvision fornece mais situações competitivas que a fonte pública. São possíveis contribuições de maior quantidade de caixas e padrões diferentes de sobreposição. Essa comparação exigiria medir as mesmas categorias para ambas as fontes; a presente reprodução instrumentada detalha o torchvision.

## Conclusão segura para apresentação

“Na MOT17-09, o torchvision tem AP50 maior, mas o baseline produz mais fragmentação de identidades. Investigamos se isso ocorria simplesmente porque caixas consecutivas da mesma pessoa caíam abaixo do limiar de associação. Não: entre os pares consecutivos que conseguimos associar ao GT, o IoU mediano foi 0,906 e nenhum ficou abaixo de 0,3. Portanto, o excesso de IDs não se explica por esse mecanismo simples. Lacunas de detecção e competição entre associações são hipóteses plausíveis, mas precisam de inspeção dos eventos para determinar sua contribuição.”

## Reprodução e limites

Os cálculos usam `data/detections/torchvision/MOT17-09`, `det/det.txt`, `gt/gt.txt` e as predições em `outputs/parte1/predictions/`. A sequência tem 525 quadros. O limiar de score é 0,5, o de associação do baseline 0,3 e o de diagnóstico GT-detecção/predição 0,5. A figura diagnóstica local está em `outputs/parte1/examples/torchvision-switch-207-208.png`.

Esta é uma análise de uma sequência e de uma execução local; não é uma nova métrica oficial nem prova geral sobre os detectores. CPU e CUDA podem causar pequenas diferenças numéricas.
