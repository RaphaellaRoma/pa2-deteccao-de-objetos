# Resultados integrados — MOT17

O split usa treino 04/05/10/11 e validação 02/09/13, sem misturar variantes
DPM/FRCNN/SDP do mesmo vídeo. A GRU aprende deslocamento da caixa por smooth L1
em trajetórias GT contíguas. Na inferência, só recebe detecções FRCNN públicas:
prevê a próxima caixa, associa por IoU/Hungarian e avança o estado durante
quadros sem associação. O baseline e a GRU usam as mesmas caixas observadas.

| Vídeo | IDF1 baseline | IDF1 GRU | IDs previstos baseline/GRU |
| --- | ---: | ---: | ---: |
| 02 | 0,3632 | 0,3764 | 135 / 123 |
| 09 | 0,5515 | 0,5043 | 50 / 44 |
| 13 | 0,4640 | 0,4592 | 518 / 457 |

A queda no número de identidades previstas não garante aumento de IDF1.
O modelo final usa `max_age=8`, enquanto o baseline usa 2. Para isolar essa
decisão, `temporal_analysis.py` avalia a mesma GRU com os dois limites:

| Vídeo | IDF1 GRU age=2 | IDF1 GRU age=8 |
| --- | ---: | ---: |
| 02 | 0,3643 | 0,3764 |
| 09 | 0,5515 | 0,5043 |
| 13 | 0,4735 | 0,4592 |

Há 222 episódios GT com `visibility=0`. Em apenas 21 havia um ID previsto
associado imediatamente antes da oclusão; a sobrevivência desse subconjunto
foi 5/21 com age=2 e 6/21 com age=8. O denominador pequeno limita conclusões.
O gráfico separa as durações e informa o número elegível em cada faixa.

Três trocas de ID foram selecionadas automaticamente das previsões, uma por
vídeo, sempre com faixa GT/predição e caixa futura GRU:

| Vídeo | Pessoa GT | Observações GT antes → depois | ID previsto antes → depois |
| --- | ---: | ---: | ---: |
| 02 | 37 | 164 → 176 | 28 → 30 |
| 09 | 22 | 234 → 246 | 11 → 24 |
| 13 | 37 | 67 → 79 | 69 → 78 |

Esses casos mostram perda de continuidade ao longo de uma lacuna de 12 quadros
entre associações GT. A visualização permite conferir se a previsão se desloca
em direção à caixa observada no retorno; a lacuna, sozinha, não prova que houve
oclusão completa em todos os quadros ou qual foi a causa única da troca. Um
prazo maior retém memória por mais tempo, mas também pode gerar associações
erradas, como indica a queda de IDF1 na 09 e na 13.

No estresse, as mesmas 36 entradas (3 vídeos × 3 seeds × controle + 3 níveis)
foram dadas aos dois rastreadores. Sem retreino, o IDF1 médio baseline/GRU foi
0,4595/0,4466 no controle, 0,3784/0,3820 no leve, 0,1845/0,2499 no médio e
0,0530/0,0750 no forte. O mAP é idêntico para os dois modelos em cada nível,
pois depende das detecções de entrada, e cai de 0,3717 a 0,0256.

As métricas são do protocolo próprio `pa2-pedestrians-v1`, não do ranking oficial
MOTChallenge. Figuras e resultados completos são regenerados por
`python run_temporal.py evaluate`, `python temporal_analysis.py` e
`python temporal_stress.py`, respectivamente.
