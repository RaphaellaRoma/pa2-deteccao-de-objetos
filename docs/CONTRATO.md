# Contrato de dados e avaliação

Configuração compartilhada: `configs/parte1.json`. Alterações depois de iniciar
o treino exigem reavaliar baseline e modelo nas mesmas condições.

## Split e fonte congelada

- Treino/desenvolvimento: MOT17-04, 05, 10, 11.
- Validação: MOT17-02, 09, 13, inteiras.
- Fonte padrão das Partes 2–5: FRCNN público, confiança >= 0,5.
- As três variantes de um vídeo pertencem ao mesmo split físico.
- Escolha das sequências definida por características, antes dos resultados.
  A validação abrange densidade, ponto de vista e movimento de câmera; não é um
  conjunto de teste independente das escolhas e correções feitas durante o PA.

## Formatos

Detecções internas: `frame,x,y,width,height,score`, seis colunas sem cabeçalho.
Predições internas: `frame,id,x,y,width,height`, seis colunas.
Coordenadas internas em pixels, xywh 0-based; quadros e IDs 1-based.
IDs são locais a cada sequência, sem associação entre vídeos.
Quadros vazios também chamam `tracker.update(frame, boxes)` com shape `(0,4)`.

O leitor MOT subtrai 1 apenas de x/y; o export MOT soma 1. Não subtrair 1 de
largura ou altura. Metadados vêm de `seqinfo.ini`. Não ordenar quadros pelo
conteúdo do detector: usar a sequência completa de 1 a seqLength.

## Avaliar um modelo temporal

```python
from mot_data import Sequence, load_config, valid_gt, evaluation_mask
from metrics import evaluate

config = load_config()
sequence = Sequence.load('data/MOT17', 'MOT17-09')
raw = sequence.ground_truth()
# predicted é a saída interna do modelo, shape (N,6), sem acesso a GT na inferência.
mask = evaluation_mask(raw, predicted[:, 0], predicted[:, 2:6], config['ignore_iou'])
metrics = evaluate(valid_gt(raw), predicted[mask], config['evaluation_iou'])
```

Mesmas detecções, sequências, limiares de avaliação e protocolo para baseline e
modelo. As regras temporais de associação/sobrevivência do modelo podem mudar,
mas devem ser registradas e atribuídas corretamente à comparação.
Não usar visibilidade ou GT para decidir associações, remover FP ou criar tracks.
Preservar previsões brutas e filtradas para auditoria e contagem.

## Protocolo próprio pa2-pedestrians-v1

- GT alvo: classe 1 e marca de avaliação > 0, sem corte de visibility.
- Ignorados: marca 0 ou classes 2, 7, 8, 12. Outros objetos não são alvos e
  previsões sobre eles não são automaticamente perdoadas.
- Em cada quadro, associar previsões aos pedestres válidos por IoU >= 0,5;
  proteger esses pares e associar sobras um-para-um aos ignorados, também a 0,5.
  Remover somente as sobras associadas a ignorados, na avaliação.
- IDF1: compatibilidade espacial por identidade em todos os quadros, depois
  atribuição global um-para-um maximizando IDTP.
- Switch: ID previsto mudou desde a última correspondência válida do GT.
- Fragmentação: uma observação GT não foi associada e uma posterior voltou a ser.
- Quantidade de identidades prevista é calculada depois do filtro de avaliação;
  também reportar contagem bruta, antes do filtro.
- Desempates de matching usam a ordem determinística das entradas e o solver
  Hungarian. Não implementamos persistência de matches do avaliador CLEAR oficial.

Não se afirma equivalência integral com MOTChallenge/COCO: protocolo próprio,
sem biblioteca pronta de métricas. Referência de classes e coordenadas:
https://arxiv.org/abs/1603.00831.

## AP de detecção

AP50 e mAP@0,50:0,95 são calculados na classe pedestrian, em caixas com score
>= 0,5 (mesmo corte do tracker). O filtro de ignorados é aplicado a 0,5 antes dos
limiares de AP. Ordenar detecções por confiança, casar apenas no mesmo quadro
com GT ainda não usado, calcular envelope de precisão e 101 recalls de 0 a 1.
Matching é por imagem; AP agrega a sequência. Nenhum ID entra neste cálculo.
Como existe somente uma classe, mAP é a média dos dez APs de IoU.
Quando não há GT, AP é indefinido (`None`); não converter para desempenho perfeito.

## Detecções alternativas para a Parte 5

O pipeline aceita uma pasta com `<MOT17-XX>.csv` no formato interno de detecção,
incluindo scores. Usar as dimensões reais da sequência ao corromper caixas.
Manter a duração completa; ausência de linhas é um quadro sem detecções.
As alterações devem depender apenas das detecções, dimensões e RNG, nunca do GT.
Mesmas caixas corrompidas serão entregues ao baseline e ao modelo final, sem treino.
O gerador sintético atual não é adequado diretamente: assume tamanho 128 x 128.

## Inferência e reprodução

O checkpoint temporal não é necessário à Parte 1. O notebook `parte1.ipynb`
executa experimentos do baseline; `inferencia.ipynb` será uma entrega separada
com o modelo temporal, vídeo e contagem. O README final também precisará incluir
um comando de treino temporal, ainda de responsabilidade da implementação do modelo.
Não copiar as funções dos módulos para dentro dos notebooks.
