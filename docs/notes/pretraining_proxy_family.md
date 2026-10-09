# Milestone 4 — Branch 2: modelos proxy

**Estado:** familia definida y verificada en CPU; preparada para el benchmark de Branch 3.  
**Fuente:** `main` en [1e7b96278a6f8be52f19c8fe09fd316d436c69e8](https://github.com/adriaflores/kimi-k3-from-scratch/commit/1e7b96278a6f8be52f19c8fe09fd316d436c69e8).  
**Protocolo recibido:** `kimi_k3_milestone4_branch1_protocolo.md`, versión 1.0, 9 de octubre de 2026.

## 1. Decisión

Usaremos dos proxies de **82,22M y 122,15M parámetros** para la exploración LR × batch y su comprobación entre escalas. Son puntos dentro de una misma familia del 300M: misma profundidad y distribución de capas, con menor anchura y menos cabezas. Conservamos el vocabulario y el routing. El tamaño definitivo se ha elegido por coherencia arquitectónica, no para alcanzar exactamente una cifra redonda.

| Modelo | Papel | Parámetros totales | Activos/token* | Sin embedding/head | Activos sin embedding/head |
| --- | --- | ---: | ---: | ---: | ---: |
| [Proxy 82M](../../configs/model/kimi_k3_proxy_82m.yaml) | Exploración principal | 82.221.540 | 33.653.220 | 66.492.900 | 17.924.580 |
| [Proxy 122M](../../configs/model/kimi_k3_proxy_122m.yaml) | Transferencia entre escalas | 122.146.637 | 47.151.437 | 102.485.837 | 27.490.637 |
| [Modelo final](../../configs/model/kimi_k3_300m.yaml) | Validación selectiva y pretraining | 294.676.808 | 100.403.528 | 263.219.528 | 68.946.248 |

\* Convención del inspector existente: todos los parámetros salvo los expertos routed fuera del top-k. Incluye router, expertos compartidos y la matriz completa del vocabulario. Es un conteo estructural, no el número literal de pesos del embedding consultados por cada token.

Los conteos se obtienen instanciando el modelo real en `meta`, con KDA chunkwise. Su constructor tiene las mismas matrices, filtros y parámetros de gate que KDA fused. No se han estimado los parámetros a partir de una regla cuadrática. El informe reproducible está en [pretraining_proxy_counts.json](pretraining_proxy_counts.json).

## 2. Cómo reducimos el modelo

| Dimensión | Proxy 82M | Proxy 122M | 300M |
| --- | ---: | ---: | ---: |
| Factor de anchura `r` | 0,5 | 0,625 | 1 |
| `hidden_size` | 320 | 400 | 640 |
| Cabezas KDA y MLA | 4 | 5 | 8 |
| Dimensión por cabeza KDA | 128 | 128 | 128 |
| MLA QK unshared / shared / V | 128 / 64 / 128 | 128 / 64 / 128 | 128 / 64 / 128 |
| Rank de Q en MLA | 64 | 80 | 128 |
| Rank de KV en MLA | 32 | 40 | 64 |
| FFN denso | 1.536 | 1.920 | 3.072 |
| Latente MoE | 160 | 200 | 320 |
| Intermedio de cada experto | 136 | 168 | 272 |

Se preservan exactamente `H*D / C = 1,6`, `q_rank / C = 0,2`, `kv_rank / C = 0,1`, `latent / C = 0,5` y `dense_intermediate / C = 4,8`. Reducir el número de cabezas mantiene el tamaño del estado de cada cabeza KDA y las dimensiones de scoring de MLA, sin cambiar simultáneamente la profundidad.

Para el intermedio de expertos del segundo proxy, el escalado exacto daría 170. Elegimos **168**, múltiplo de ocho, para las matrices BF16 de grouped GEMM: una desviación del −1,18%. El proxy pequeño conserva la proporción exacta con 136. Esto es alineación de dimensiones, no una nueva optimización del kernel.

También se mantienen:

- 13 capas: `[kda,kda,kda,mla,kda,kda,kda,mla,kda,kda,kda,mla,mla]`, nueve KDA y cuatro MLA.
- Una primera FFN densa, doce MoE y bloques AttnRes de tamaño dos.
- 64 expertos routed, top-2, dos shared, QB, renormalización y normalización latente.
- Vocabulario 49.152 y tying de embedding/head; compatibilidad con los shards FineWeb-Edu existentes.
- KDA fused, chunk 64, conv kernel cuatro y gate lower bound −5; gates de salida KDA y MLA.
- RMS epsilon, inicialización nominal `0.02`, SiTU `(4,25)` y `routed_scaling_factor=1` del modelo final.

No se modifica el código del backbone, del optimizador ni del trainer. Los YAML son configuraciones completas admitidas por `KimiK3Config`; no son parches parciales que dependan de defaults.

## 3. Qué comparabilidad conservamos y qué cambia

Conservar capas y expertos facilita interpretar los resultados: la prueba no cambia la mezcla KDA/MLA, la frecuencia de MoE, la profundidad de AttnRes ni el número de estimaciones QB por token. A igual microbatch y contexto, las asignaciones por experto tienen la misma escala nominal `b*T*top_k/E`; sus valores reales dependen del routing aprendido.

Esta familia no conserva todas las proporciones. Al mantener la dimensión por cabeza, el rango del gate KDA `Wad` permanece en 128 y aumenta respecto a `C` en los proxies; también la dimensión shared de MLA permanece en 64. Estas decisiones conservan los mecanismos por cabeza, a cambio de esas proporciones relativas. No presentamos los proxies como réplicas homotéticas perfectas.

El vocabulario fijo hace que embedding/head represente el 19,13% de los parámetros del proxy pequeño, el 16,10% del intermedio y el 10,68% del final. Por eso se registran por separado parámetros totales, activos y sin embedding/head. Reducir el vocabulario para mejorar estas proporciones cambiaría datos y tarea; lo conservamos.

Tampoco introducimos μP ni un reescalado nuevo de la inicialización. Muon conserva su factor de geometría `0.2*sqrt(max(rows,cols))`, aplicado a las vistas y splits existentes. Sus matrices tienen otras anchuras y ranks: **un LR nominal ganador en un proxy no queda validado en el siguiente tamaño**. La transferencia se comprobará en Branches 5–6, tal como exige Branch 1.

## 4. Coste aproximado

El [inspector de proxies](../../scripts/inspect_pretraining_proxies.py) cuenta los productos de las proyecciones densas, expertos top-k y shared, convoluciones, núcleo KDA y atención MLA. Una multiplicación-acumulación cuenta como dos FLOPs. La matriz tied se cuenta una vez en parámetros y una vez como proyección de salida en cómputo; el lookup de entrada no se convierte en otra proyección densa.

Para KDA usamos una aproximación explícita de la aritmética WY de la referencia: por cabeza y token `6*D² + 8*chunk*D + chunk²`, con padding del último chunk. Incluye los productos principales y una estimación del solve triangular; no reproduce instrucción por instrucción FLA. MLA se estima con los productos de atención cuadrados completos: `2*H*T*(Dqk+Dv)` por token, sin descontar la mitad causal.

Se aproxima forward+backward como tres veces forward. **Se excluyen** recomputación específica de los kernels, operaciones pointwise/norm/softmax, quantiles, sorting/routing, copias, compilación y optimizador. El trabajo detached de QK-Clip se registra aparte mediante una cota de scoring cuadrado completo, sin multiplicarlo por tres. Son estimaciones para describir escala; no son un presupuesto IsoFLOPs exacto ni una medida de rendimiento.

| Modelo | Train GFLOPs/token aprox., T=128 | Respecto al 300M | Pesos FP32 | Pesos + grads + estados* |
| --- | ---: | ---: | ---: | ---: |
| Proxy 82M | 0,2238 | 0,346 | 313,65 MiB | 0,978 GiB |
| Proxy 122M | 0,3102 | 0,480 | 465,95 MiB | 1,439 GiB |
| 300M | 0,6462 | 1 | 1.124,10 MiB | 3,411 GiB |

`T=128` es una referencia de cálculo, **no la elección del contexto del estudio**. El JSON incluye también 64, 256 y 2048 para mostrar su dependencia.

\* Cota de memoria persistente con todos los gradientes y estados asignados: `12*P_Muon + 16*P_AdamW` bytes. Incluye pesos/grads FP32, momentum Muon y dos momentos AdamW. No incluye activaciones, stacks de expertos, workspaces de Muon/FLA, allocator ni runtime CUDA; **no es el pico de VRAM**. La actividad dispersa puede reducir estados inicialmente.

Los 64 expertos y las 13 capas conservan costes de gestión aun al reducir matrices. Branch 3 debe medir throughput y VRAM reales. No inferimos velocidad a partir de estos FLOPs ni del número de parámetros.

## 5. Restricciones de implementación revisadas

Se revisaron los constructores, `parameter_groups.py`, `qk_clip.py`, el CLI y las dependencias fijadas por `uv.lock`: PyTorch 2.11.0+cu128, FLA/fla-core 0.5.2 y triton-windows 3.6.0.post26.

FLA 0.5.2 admite chunk 32/64, safe gate en `[-5,0)` y KDA key dimension hasta 256. Los tres modelos conservan chunk 64 y dimensión 128. Q y V tienen el mismo número de cabezas; la condición de agrupación GVA se satisface también con cinco cabezas.

En [GroupedMMUtils.h de PyTorch 2.11](https://github.com/pytorch/pytorch/blob/v2.11.0/aten/src/ATen/native/GroupedMMUtils.h), los strides relevantes de grouped GEMM requieren alineación de 16 bytes: ocho elementos BF16. Nuestros latentes e intermedios son múltiplos de ocho; los stacks temporales contiguos y sus transpuestos siguen el camino ya integrado. La ejecución CUDA y su rendimiento requieren la comprobación real de Branch 3.

El head packing de KDA sigue siendo `(3,H,128,C)`; MLA Q es `(H,192,q_rank)` y KV es `(H,256,kv_rank)`, con splits independientes de K/V `(128,128)`. Embedding/head continúa en AdamW y se deduplica; gates y latentes conservan su asignación existente.

## 6. Verificación realizada

En este entorno hay CPU y **no hay CUDA**. Se instaló PyTorch **2.11.0+cpu** para verificar la misma versión del framework. Se instanciaron los proxies con las dimensiones completas, las 13 capas, todos los expertos y el vocabulario completo. Para ejecutar en CPU se cambió únicamente el backend KDA a chunkwise mediante una copia temporal de la configuración; los YAML entregados mantienen fused.

Ambos proxies superaron el [checker reproducible](../../scripts/check_pretraining_proxies.py):

- Dos updates BF16/autocast con almacenamiento FP32, `b=1`, `T=64`, acumulación dos y Muon batched.
- Forward/backward, gradientes y parámetros finitos, sin skips y con 256 tokens contabilizados correctamente.
- Observación QK-Clip de las 16/20 cabezas MLA por update; en estos batches no se activó clipping QK.
- Métricas MoE de las doce capas: 128 tokens y 256 asignaciones por capa/update, con biases QB finitos.
- Checkpoint tras el primer update: recuperación de un sample de pesos y de todos los biases QB deliberadamente alterados, paso, cursor y RNG; continuación hasta el segundo update.
- Evaluación finita que mantiene QB congelado y restaura el modo de entrenamiento.

Son **tokens sintéticos y un scheduler de plumbing**: sus losses, LR `3e-4`, warmup cero y cosine de dos pasos no son resultados de investigación ni valores elegidos para el estudio. El checker no sustituye un estudio de estabilidad, ni afirma equivalencia bit a bit frente a una trayectoria ininterrumpida de estos proxies. El registro escalar está en [pretraining_proxy_checks.json](pretraining_proxy_checks.json).

Además, se ejecutaron las pruebas existentes de parameter groups, Per-Head Muon y entrenamiento con QK-Clip: **65 passed, 1 skipped**. La omitida requiere CUDA. Esas pruebas incluyen clipping activo y comprobaciones numéricas de las vistas por cabeza y de resume. No se ha ejecutado toda la suite del repositorio.

## 7. Uso y traspaso a Branch 3

Desde la raíz del repositorio:

```bash
uv run python scripts/inspect_pretraining_proxies.py \
  configs/model/kimi_k3_proxy_82m.yaml \
  configs/model/kimi_k3_proxy_122m.yaml \
  configs/model/kimi_k3_300m.yaml \
  --seq-len 64 128 256 2048
```

El checker sirve para repetir la comprobación de plumbing en la 4050 conservando los YAML fused:

```bash
uv run python scripts/check_pretraining_proxies.py \
  configs/model/kimi_k3_proxy_82m.yaml \
  configs/model/kimi_k3_proxy_122m.yaml \
  --device cuda --precision bf16 --muon-compile \
  --output results/training/proxy_cuda_check.json
```

Este comando hace una prueba corta sintética, no mide throughput estable. Branch 3 puede ajustar sus dimensiones de prueba al microbatch/contexto que investigue y debe comprobar el coste de compilación de las nuevas formas.

Branch 3 recibe la familia **82M → 122M → 300M** y debe:

1. Confirmar ambos proxies con KDA fused, grouped GEMM y Muon compiled/batched en la 4050.
2. Elegir un contexto y microbatch comunes si son viables, y medir throughput, pico de VRAM y overhead de compilación/evaluación.
3. Fijar el grid LR × acumulación y `D/W/V/E/L`, con la conversión exacta a pasos establecida en Branch 1.
4. Repartir el objetivo orientativo de 8–12 horas incluyendo margen y validación. Si no cabe, reducir runs o rango antes de cambiar la familia; cualquier rediseño debe quedar registrado antes del estudio.

Branch 2 termina con las configuraciones y su verificación estructural/CPU. La compatibilidad medida de kernels CUDA y la viabilidad temporal quedan como condiciones de entrada al grid de Branch 4, responsabilidad de Branch 3.
