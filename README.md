# PA2 - Identidade ao longo do tempo

Rastreamento próprio de pedestres com MOT17 e detector pré-treinado torchvision.
Associação, gestão de tracks, NMS e métricas implementados no repositório.

## Estado do trabalho

- Parte 0 implementada: gerador, oclusão real, detector simulado e testes.
- Parte 1: pipeline, notebook e avaliação pública nas sete sequências executados.
  Detector pré-treinado testado em três quadros de cada vídeo de validação.
  Comparação final exige terminar a inferência dos 1.875 quadros de validação.
- Partes 2–4 (modelo temporal, treino, ablação e memória) ainda pendentes.
- Parte 5: entrada de detecções alternativas pronta; estresse ainda pendente.
- Checkpoint temporal e `inferencia.ipynb` ainda pendentes. `parte1.ipynb` é o
  notebook de experimentos do baseline e não substitui a inferência final.

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

**`parte1.ipynb`** chama os mesmos módulos. Download, benchmark e inferência
pesada estão desligados por padrão: alterar `RUN_DOWNLOAD`, `RUN_BENCHMARK` e
`RUN_DETECT` na primeira célula. Cache parcial não é apresentado como comparação
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

Dados, pesos, caches, vídeos, resultados volumosos, PDF e conversas não são
publicados. O `AI_LOG.md` técnico exigido na entrega será revisado para descrever
uso de IA sem histórico de conversas ou informações pessoais.
