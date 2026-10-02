# PA2 - Identidade ao longo do tempo

Rastreamento próprio de pedestres com MOT17 e detector pré-treinado torchvision.
Associação, gestão de tracks, NMS e métricas implementados no repositório.

## Notebook da apresentação

**[experimentos_pa2.ipynb](experimentos_pa2.ipynb)** é o notebook do trabalho,
organizado das Partes 0 a 5, com métricas, figuras e interpretação dos
resultados. A Parte 0 é uma referência breve; as seções 2–4 mostram os resultados
do experimento completo em **[trabalhop2.ipynb](trabalhop2.ipynb)**.
O código pesado permanece nos módulos `.py`. As saídas estão
salvas; para reproduzir, executar todas as células com os dados e caches
preparados. Inferência do detector e benchmark ficam opcionais na seção da
Parte 1. A Parte 5 compara o baseline e o checkpoint temporal congelado.

Os notebooks individuais permanecem como versões por parte; use o notebook
unificado para a apresentação. **[inferencia.ipynb](inferencia.ipynb)** recebe
uma sequência MOT17 e produz MP4 com IDs e contagem única.

## Estado do trabalho

- Parte 0 implementada: gerador, oclusão real, detector simulado e testes.
- Parte 1 executada: avaliação pública nas sete sequências e comparação com
  torchvision nos 1.875 quadros completos de validação. Resultados e figuras
  registrados no notebook, com `complete_part1=True`.
- Parte 2: GRU causal treinada em trajetórias MOT17 04/05/10/11; comparação com o
  baseline nas sequências 02/09/13 usando as mesmas detecções públicas.
- Parte 3: ablação sintética RNN/GRU/LSTM, janelas 4/8/16/32 e três seeds no
  notebook de experimentos; checkpoint RNN MOT17 para horizonte analítico.
- Parte 4: gradientes dos modelos treinados, sobrevivência por oclusão GT, três
  faixas de falhas reais e intervenção `max_age=2` para 8.
- Parte 5: três intensidades e três seeds, sem retreino; mAP e IDF1 lado a lado
  para baseline e modelo temporal em 02/09/13.
- Checkpoints em `checkpoints/temporal_mot17.pt` e
  `checkpoints/temporal_rnn_mot17.pt`; `inferencia.ipynb` e `render_video.py`
  produzem vídeo e contagem.

## Ambiente

Python 3.12. Criar o ambiente e instalar dependências de notebook/detector:

```powershell
py -3.12 -m venv .venv
.venv\Scripts\python -m pip install -r requirements-detector.txt
```

Nos comandos abaixo, `python` significa o Python desse ambiente. No Windows,
use `.venv\Scripts\python` se o ambiente não estiver ativado. Para somente a
Parte 0, basta `requirements.txt`. Selecionar o mesmo ambiente no kernel Jupyter.

Torch 2.11.0 e torchvision 0.26.0. Para GPU, instalar o build CUDA compatível
com o [seletor oficial do PyTorch](https://pytorch.org/get-started/locally/),
mantendo as versões. `--device auto` usa CUDA quando disponível. `--device cuda`
falha explicitamente sem suporte instalado. A arquitetura não muda com o hardware.

No Colab, clonar o repositório, instalar dependências e abrir o mesmo notebook
na raiz. Os módulos continuam em arquivos Python, sem duplicação nas células.

## Parte 0

**`parte0.ipynb`** reúne a apresentação da Parte 0: contrato do gerador,
oclusão real com figura e animação, detector simulado, casos manuais das métricas,
piso fácil, curvas de dificuldade em três seeds e testes automatizados.
As saídas estão salvas no notebook. Executar todas as células regenera os
artefatos; o código de experimentos permanece em `run_synthetic.py`, função
`run_experiments`, também chamada pelo comando abaixo.

```powershell
python run_synthetic.py
```

Saídas em `outputs/parte0/`: resultados JSON, animação e figura de oclusão,
gráfico de dificuldade. Piso fácil: IDF1=1, zero switches, 5/5 IDs. Oclusão de
10 quadros verificada nos pixels. Experimentos usam seeds 0, 1 e 2.
O sintético exclui observações totalmente ocultas; o protocolo MOT não faz isso.

## Parte 1: dados

Fonte oficial: [MOT17](https://motchallenge.net/data/MOT17/).
Labels e detecções públicas (aproximadamente 10 MB):

```powershell
python download_mot17.py --package labels
```

Imagens dos três vídeos de validação, seletivamente via HTTP Range:

```powershell
python download_mot17.py --package frames --sequences MOT17-02 MOT17-09 MOT17-13
```

Usa curl/IPv4 para download retomável; extrai somente treino/FRCNN, evitando
cópias DPM/SDP. Valida CRCs ao extrair arquivos novos. Blocos remotos persistem
em `data/archives/ranges/`. Alternativa se o servidor não suportar Range:
`--package full`, que baixa o ZIP inteiro (aproximadamente 5,5 GB) e extrai as
sequências pedidas. `--root` permite indicar dados já baixados.

Estrutura: `data/MOT17/train/MOT17-XX-FRCNN/{seqinfo.ini,gt/gt.txt,det/det.txt,img1/}`.
Dados não são incluídos no Git.

## Parte 1: executar e avaliar

Avaliação pública, sem imagens ou GPU:

```powershell
python parte1.py evaluate --sources public
```

Benchmark antes da inferência completa (cinco warmups e 30 quadros):

```powershell
python parte1.py benchmark --device auto
```

Um comando executa o detector nos três vídeos e avalia ambas as fontes:

```powershell
python parte1.py run --device auto
```

Para separar extração e avaliação, inclusive em duas máquinas:

```powershell
python parte1.py detect --device auto
python parte1.py evaluate --sources public torchvision
```

O segundo comando é o comando único de avaliação depois de preparar os caches.
Não há treino de detector nessa parte; o comando de treino temporal será
adicionado quando o modelo temporal estiver implementado.

**`parte1.ipynb`** chama os mesmos módulos. A versão executada habilita benchmark
e inferência pela primeira célula (`PA2_RUN_BENCHMARK` e `PA2_RUN_DETECT`). Para
apenas avaliar caches, defina essas variáveis como `"0"` nessa célula. Download
continua opcional via `RUN_DOWNLOAD`. Cache parcial não é apresentado como comparação
completa. Caches completos podem ser avaliados sem GPU.

Detector: Faster R-CNN ResNet-50 FPN, COCO_V1, classe `person`, sem fine-tuning.
NMS interno do RPN e caixas finais substituído pelo próprio em contexto
temporário e restaurado após erro. Redimensionamento e limites de propostas
seguem o modelo original. NMS roda em CPU mesmo com rede em CUDA; o benchmark
inclui esse custo.

Cache por quadro em `data/detections/torchvision/`, com hashes de imagens,
detecções, código e configuração, incluindo quadros vazios. Retomável após
interrupção; rejeita mudanças incompatíveis. Builds +cpu/+cu da mesma versão
podem retomar o cache; dispositivo e versão exata registrados por quadro.
Pequenas diferenças numéricas CPU/GPU podem existir. Lock impede duas inferências
simultâneas escrevendo no mesmo cache.

Saídas em `outputs/parte1/`:

- `metrics.csv` e `results.json`: métricas, configuração, versões e hashes.
  `complete_part1` indica ambas as fontes nos três vídeos inteiros de validação.
- `predictions/<source>/<sequence>.txt`: saída bruta em formato MOT, inclusive
  caixas ignoradas somente na avaliação.
- `descolamento.png`: gráfico obrigatório na validação.
- `descolamento_todas.png`: também inclui desenvolvimento, identificado na tabela.
- `benchmark.json`: tempo, memória e projeção de custo.
- `run-status.json`: detecting/evaluating/complete/partial/failed no comando `run`.

Gráfico: mAP/IDF1 em cima; razão de IDs e switches/ID GT embaixo, com eixos
distintos identificados. Ordenação por densidade GT. Fonte única ou sequência
parcial é rotulada como comparação pendente/depuração.

Exemplos visuais de manutenção, troca e retorno com ID novo:

```powershell
python parte1.py visualize --sequence MOT17-09
```

Eventos encontrados nas previsões; categorias inexistentes não são inventadas.

## Resultados públicos executados

FRCNN público, score >= 0,5, associação IoU >= 0,3, max_age=2:

| Sequência | Uso | AP50 | mAP 0,50:0,95 | IDF1 | IDs previstos / GT |
| --- | --- | ---: | ---: | ---: | ---: |
| MOT17-04 | Treino | 0,5446 | 0,4334 | 0,5716 | 149 / 83 |
| MOT17-05 | Treino | 0,5045 | 0,3235 | 0,5461 | 160 / 133 |
| MOT17-10 | Treino | 0,5810 | 0,4029 | 0,4448 | 329 / 57 |
| MOT17-11 | Treino | 0,5840 | 0,4906 | 0,5525 | 115 / 75 |
| MOT17-02 | Validação | 0,3363 | 0,2659 | 0,3632 | 135 / 62 |
| MOT17-09 | Validação | 0,5544 | 0,4638 | 0,5515 | 50 / 26 |
| MOT17-13 | Validação | 0,5607 | 0,3853 | 0,4640 | 518 / 110 |

Resultados do protocolo próprio, não oficiais. AP e IDF1 têm definições distintas;
suas magnitudes não são percentuais diretamente equivalentes. Densidade não
explica tudo: a sequência 13, com câmera móvel, produz muito mais IDs que a 09.

Benchmark executado em CPU/MOT17-09: cerca de 3,36 s/quadro, projeção de 105
minutos para 1.875 quadros, pico do processo de aproximadamente 1.053 MB.
São medidas daquela execução, não garantia para outro hardware ou vídeo.

## Comparação completa da Parte 1

Valores da execução CUDA preservada em `parte1.ipynb`, com os mesmos limiares
da tabela anterior e todas as sequências de validação completas:

| Sequência | Fonte | AP50 | mAP 0,50:0,95 | IDF1 | IDs previstos / GT |
| --- | --- | ---: | ---: | ---: | ---: |
| MOT17-02 | Público | 0,3363 | 0,2659 | 0,3632 | 135 / 62 |
| MOT17-02 | Torchvision | 0,4166 | 0,2338 | 0,3093 | 704 / 62 |
| MOT17-09 | Público | 0,5544 | 0,4638 | 0,5515 | 50 / 26 |
| MOT17-09 | Torchvision | 0,7160 | 0,4319 | 0,4511 | 326 / 26 |
| MOT17-13 | Público | 0,5607 | 0,3853 | 0,4640 | 518 / 110 |
| MOT17-13 | Torchvision | 0,5432 | 0,2628 | 0,3337 | 1249 / 110 |

Na MOT17-09, AP50 cresce 0,1616, mas IDF1 cai 0,1004 e a contagem sobe
de 50 para 326 IDs para 26 pessoas. Na MOT17-02 ocorre a mesma direção no
AP50 e no IDF1. Caixas melhores no limiar IoU 0,5 não asseguram identidade
consistente com associação pela última caixa observada. O mAP cai nas três
sequências; o ganho de AP50 não representa melhoria em todos os limiares.

O público obtém maior IDF1 nas três sequências. Mantemos o FRCNN público já
fixado no contrato para comparar baseline e modelo temporal com entradas
iguais. A MOT17-13 tem menor densidade que a 02, mas gera mais identidades
excedentes. Movimento, oclusão e variação das caixas são hipóteses a investigar
nas figuras; estas métricas agregadas não isolam suas causas.

Benchmark CUDA: 0,3802 s/quadro, projeção de 11,88 minutos para 1.875 quadros,
pico de 671,3 MB na GPU e 1.649,9 MB no processo. A projeção não é o tempo
medido de uma execução completa. A execução local em CPU também terminou;
pequenas diferenças numéricas não mudaram a ordenação de IDF1 entre fontes.

## Regras e integração

Contrato em **[docs/CONTRATO.md](docs/CONTRATO.md)** e `configs/parte1.json`.

- Split: treino 04/05/10/11; validação 02/09/13; variantes não cruzam splits.
- FRCNN público congelado para baseline e modelo temporal.
- GT: pedestres válidos sem corte de visibility.
- Protocolo `pa2-pedestrians-v1`: priorizar GT válido, ignorar sobras associadas
  a anotações ignoradas apenas na avaliação; nunca fornecer GT ao tracker.
- Baseline: última caixa observada, Hungarian, nascimento imediato e tolerância
  de dois quadros intermediários; não prevê movimento.
- IDF1 global, switches desde última associação, fragmentação após lacuna nas
  observações GT; contagem antes e depois do filtro.
- AP50 e mAP em dez IoUs, 101 recalls, matching por imagem e agregação por vídeo,
  classe única e score >= 0,5. Sem equivalência integral ao protocolo COCO/MOT.

Entrada para estresse: pasta com CSVs internos `frame,x,y,width,height,score`:

```powershell
python parte1.py evaluate --sources external --split validation --detections-dir data/stress --output outputs/stress
```

Depuração: `--max-frames 3` e cache/saída separados. Resultados parciais não
substituem vídeos completos.

## Partes 2–4: movimento, ablação e memória

Treino e comparação da GRU na trilha A, com split por vídeo físico e entrada
FRCNN pública congelada:

```powershell
python run_temporal.py all --device auto
python run_temporal.py train --cell RNN --checkpoint checkpoints/temporal_rnn_mot17.pt
python temporal_analysis.py --device auto
```

O tracker prevê a caixa seguinte, associa por Hungarian/IoU e mantém o estado
recorrente durante lacunas. Ele não recebe GT na inferência. O modelo prevê
deslocamento residual sobre a caixa corrente, treinado com smooth L1 em janelas
contíguas de GT dos vídeos 04/05/10/11. As previsões são avaliadas nos vídeos
02/09/13 completos. O baseline tem `max_age=2`; o modelo final usa 8. A análise
da Parte 4 também compara 2 e 8 no mesmo checkpoint.

| Sequência | IDF1 baseline | IDF1 GRU | IDs baseline / GRU |
| --- | ---: | ---: | ---: |
| MOT17-02 | 0,3632 | 0,3764 | 135 / 123 |
| MOT17-09 | 0,5515 | 0,5043 | 50 / 44 |
| MOT17-13 | 0,4640 | 0,4592 | 518 / 457 |

O efeito é misto: a GRU reduz o número de IDs criados nas três sequências,
mas só aumenta IDF1 na 02. A análise contém 222 episódios GT de visibilidade
zero; só 21 tinham ID associado imediatamente antes, denominador da taxa de
sobrevivência. Com `max_age=2`, 5/21 mantiveram o ID; com 8, 6/21. A extensão
do prazo melhora a 02, porém diminui IDF1 na 09 e na 13. Três faixas reais de
troca, curvas de gradiente RNN/GRU e o gráfico por duração estão em
`outputs/temporal_analysis/`. A ablação RNN/GRU/LSTM de três seeds e quatro
janelas permanece no fluxo original de `trabalhop2.ipynb`, com tabela exportada
para `outputs/ablation/idf1.csv` quando executada na GPU.

Inferência de uma sequência completa com vídeo e contagem única:

```powershell
python render_video.py data/MOT17/train/MOT17-09-FRCNN --checkpoint checkpoints/temporal_mot17.pt --output outputs/inferencia/MOT17-09.mp4 --device auto
```

O mesmo fluxo está em `inferencia.ipynb`. A contagem corresponde a IDs
previstos únicos e pode exceder o número de pessoas quando ocorre fragmentação.

## Parte 5: qualidade do detector

```powershell
python temporal_stress.py --device auto
```

Usa as detecções públicas FRCNN aceitas pelo corte 0,5, três vídeos completos
de validação e seeds 0/1/2. Inclui controle original e três intensidades em
`configs/parte5.json`: descarte de 10/30/50%, ruído relativo de 3/8/15% e
Poisson de 0,5/1,5/3 falsos positivos por quadro. As intensidades combinam as
três alterações; não isolam o efeito individual de cada uma.

O ruído gaussiano desloca centros em unidades de largura/altura e altera
tamanhos multiplicativamente (log-normal). Caixas corrompidas ficam dentro
da imagem. FP têm posição uniforme e tamanhos/scores amostrados das detecções
originais. FP significa injeção sem correspondência conhecida: por acaso pode
sobrepor uma pessoa real. Scores originais não mudam. O gerador não recebe GT.

Saídas em `outputs/parte5/`: `detections/<nivel>/seed-<seed>/<sequencia>.csv`,
previsões MOT, `metrics.csv`, `results.json` com hashes e `estresse.png`.
O gráfico mostra média com peso igual por sequência; barras representam o
desvio padrão das médias entre seeds, não intervalo de confiança.
`parte5.ipynb` apresenta o experimento sem duplicar funções.

O comando gera as entradas determinísticas, executa o checkpoint congelado
sobre todos os CSVs e avalia ambos os rastreadores (`complete_part5=True`).
Hashes permitem conferir as entradas. Não há retreino nem uso de GT na
inferência.

Baseline executado em 36 avaliações completas (três sequências, três seeds,
controle e três intensidades). Médias com peso igual por sequência:

| Intensidade | mAP 0,50:0,95 | IDF1 baseline | IDF1 GRU |
| --- | ---: | ---: | ---: |
| Original | 0,3717 | 0,4595 | 0,4466 |
| Leve | 0,2764 | 0,3784 | 0,3820 |
| Médio | 0,1165 | 0,1845 | 0,2499 |
| Forte | 0,0256 | 0,0530 | 0,0750 |

Médias com peso igual para as três sequências e três seeds. A GRU melhora o
IDF1 médio nas três intensidades corrompidas, mas fica abaixo do baseline no
controle original. O mAP é o mesmo para ambos porque recebem as mesmas caixas.

## Testes e arquivos

```powershell
python -m unittest discover -s tests -v
```

Testes: IDF1, switches, fragmentações, NMS, oclusão por pixels, conversão MOT,
ignorados, ausência de vazamento de GT, AP, cache retomável/corrompido, lock,
memória e forward real do detector com NMS proibido bloqueado. Notebook
verificado com fixture e resultados iguais ao script.

Parte 0: `synthetic.py`, `geometry.py`, `baseline.py`, `metrics.py`, `run_synthetic.py`.
Parte 1: `mot_data.py`, `detection_metrics.py`, `detector.py`, `parte1.py`,
`visualization.py`, `download_mot17.py`, `parte1.ipynb`.

Dados MOT17, caches, vídeos, resultados volumosos, PDF e conversas não são
publicados. Os dois checkpoints pequenos e reproduzíveis são incluídos. O
`AI_LOG.md` local deve ser revisado pelos integrantes antes da entrega final;
não contém histórico de conversas.
