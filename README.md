# PA2 - Identidade ao longo do tempo

Implementação incremental do assignment de Aprendizado Profundo sobre identidade ao longo do tempo.
Entrega indicada: **02/10, 23h59**, com apresentação e trabalho em dupla.

## Estado atual

Parte 0 implementada e executada: vídeos sintéticos 128 x 128, oclusão por
profundidade, detector corrompível, métricas próprias e baseline IoU/Hungarian.
**O trabalho completo ainda não está pronto.** Não há modelo treinado, resultados
MOT17, checkpoint ou notebook de inferência nesta etapa.

## Executar

Windows com Python 3.12:

```powershell
py -3.12 -m venv .venv
.venv\Scripts\python -m pip install -r requirements.txt
.venv\Scripts\python -m unittest discover -s tests -v
.venv\Scripts\python run_synthetic.py
```

Linux/Colab: use `python` no lugar de `.venv\Scripts\python` após instalar
`requirements.txt`. Os experimentos da Parte 0 não precisam de PyTorch ou GPU.

Saídas em `outputs/parte0/`:

- `results.json`: métricas e seeds de todos os experimentos.
- `oclusao.gif`: animação com desaparecimento real de um objeto atrás de outro.
- `oclusao.png`: antes, durante e depois de 10 quadros de oclusão completa.
- `baseline_dificuldade.png`: variações de quantidade, velocidade e oclusão,
  com média e desvio amostral entre seeds 0, 1 e 2.

## Convenções e resultados verificados

Caixas usam `(x, y, largura, altura)`. Quadros e IDs começam em 1.
O baseline usa Hungarian com IoU mínimo 0,3 e a última caixa observada.
Pares abaixo do limiar são excluídos do objetivo antes da atribuição; maximiza-se
primeiro a cardinalidade e depois a soma de IoU. Cada detecção sem par inicia
um ID; uma track permite até 2 quadros intermediários sem observação. Tracks
ocultas persistem na memória, mas não emitem caixas na saída do baseline.

IDF1 usa atribuição global um-para-um entre identidades, contando coincidências
espaciais com IoU >= 0,5 ao longo da sequência. ID switches comparam com o último
ID associado, inclusive após lacunas. Fragmentações contam retorno após uma
observação GT sem correspondência. Uma mudança de ID sem lacuna é um switch,
mas não uma fragmentação. Erro de contagem é a diferença absoluta de IDs únicos.

Casos analíticos de teste (duas identidades por quatro quadros):

| Caso | IDF1 | ID switches | Fragmentações |
| --- | --- | --- | --- |
| Predição perfeita | 1,00 | 0 | 0 |
| Duas identidades trocadas na segunda metade | 0,50 | 2 | 0 |
| Uma identidade dividida na segunda metade | 0,75 | 1 | 0 |

O piso fácil obteve IDF1 = 1,0, zero switches e contagem 5/5. Os 13 testes
automatizados passaram. Maior velocidade derruba o baseline; aumentar apenas
a quantidade, neste cenário lento e espaçado, não produziu degradação.

O GT sintético preserva caixas amodais e fração de pixels visíveis, calculada
depois de desenhar todos os objetos. O simulador só detecta objetos com algum
pixel visível. A avaliação sintética também exclui observações totalmente
ocultas. Essa convenção não deve ser confundida com o protocolo oficial MOT17:
filtragem de classes e tratamento de regiões ignoradas ainda precisam ser feitos.
As oclusões controladas são artificiais; não representam a distribuição real.

## Próximas etapas do enunciado

- Parte 1: obter anotações e detecções MOT17; split por sequência física, sem
  separar cópias DPM/FRCNN/SDP de uma mesma sequência entre treino e validação;
  adicionar detector torchvision, mAP e gráfico de descolamento em dois painéis.
- Parte 2: proposta inicial de trilha A, GRU de movimento com estado por track,
  Smooth L1 e rollout durante oclusão; congelar a fonte de detecções.
- Parte 3: proposta de eixo 2, teacher forcing/scheduled sampling/free-running
  com clipping ligado/desligado e três seeds em cada configuração.
- Parte 4: três falhas, curva de gradiente por lag, horizonte empírico e uma
  correção avaliada antes/depois nas mesmas condições.
- Parte 5: proposta de degradação do detector em três intensidades, sem retreino.
- Entrega: comandos únicos de treino e avaliação, `inferencia.ipynb`, checkpoint
  e material reproduzível para apresentação. Não há relatório escrito exigido.

Fonte de dados indicada no enunciado: https://motchallenge.net/data/MOT17/.
Começar pelo pacote de anotações; os frames serão necessários para a comparação
com detector torchvision e a galeria visual. Download ainda não realizado.

## Autoria e restrições

Sem rastreadores ou métricas de tracking prontos, nem `torchvision.ops.nms`.
`scipy.optimize.linear_sum_assignment` é usado somente como solver Hungarian;
geometria, associação, gestão de tracks, NMS e métricas são implementados aqui.
O registro técnico de uso de IA (`AI_LOG.md`), exigido na entrega, será incluído
após revisão, sem histórico de conversas ou informações pessoais.
