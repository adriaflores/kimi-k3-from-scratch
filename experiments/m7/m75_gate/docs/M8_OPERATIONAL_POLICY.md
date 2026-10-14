# Propuesta operativa para M8 — pendiente de GO del gate real M7.5

## Invariantes

- Continuar desde `/workspace/kimi/runs/m73/m73_pilot_80m_v1_20261014T065121Z/last.pt`, no desde M6 qualification ni desde pesos aleatorios. Conservar siempre el original como *source checkpoint* inmutable y el respaldo local con SHA `cd72de269f7036b7a96ffe6f291339e3743eac90a98322d7dea20b9691e17e12`.
- Receta: LR base 0.003, cosine hasta `decay_steps=183106`, warmup 1832, min LR ratio 0.1, BF16/FP32 weights, batch 32×1024 y 1 acumulación, Muon compilado+batched y AdamW fused. Estado completo de optimizer, RNG, loaders y step cargado por la rutina oficial.
- M8 produce un **nuevo `run_dir`** y **nuevo `last.pt`**, sin apuntar la salida al checkpoint de M7.3. Si se reanuda M8 tras interrupción, usar el checkpoint M8 más reciente con recibo válido.
- M8 debe usar el mismo `/workspace/kimi/reference` con hashes del manifest M7.3, mismo Python/pip y el H100 NVL. `OMP_NUM_THREADS=8` y `MKL_NUM_THREADS=8` preservan el parámetro `num_threads` guardado.

## Cadencias de la receta propuesta

| Señal | Intervalo | Tiempo aproximado según 64.194 tok/s |
|---|---:|---:|
| train logging + MoE/QK diagnostics | 64 updates | ~33 s |
| validation, 8 batches = 262.144 tokens de prefijo val | 512 updates | ~4,4 min |
| checkpoint `last.pt` | 4096 updates | ~35 min |
| respaldo externo | cada 4 checkpoints como máximo | ~2,3 h |

**Nota:** los intervalos se aplican a **steps globales** (incluyen 2.442 de M7.3). El primer checkpoint periódico de M8 será al step 4096; además, el trainer guarda al finalizar en 183106.

## Backups: protocolo necesario ANTES de lanzar M8

1. Al producirse `event=checkpoint`, el trainer ya ha cerrado y reemplazado `last.pt` desde su fichero temporal. Un watcher externo debe detectar **solo eventos completos** y copiar el fichero a `snapshot_step_<STEP>.pt.tmp`, renombrarlo atómicamente y emitir SHA+tamaño/step. Copiar `last.pt` al detectar su timestamp directamente es más frágil.
2. Retener en remoto **3 snapshots completos verificados** como mínimo, además de `last.pt` activo y el checkpoint fuente M7.3. A ~2,49 GB cada uno, 3 snapshots ≈7,46 GB, más el `last.pt` M8 y fuente ≈4,97 GB. Verificar espacio libre antes de iniciar; evitar guardar todos los checkpoints sin rotación.
3. Copia **fuera de Vast** a intervalos no mayores que 4 checkpoints, más el checkpoint final, con SHA-256 recalculado en destino y recibo `verified=true`. No borrar la última copia local válida. La retención/remoción de snapshots antiguos debe condicionarse a SHA verificado y disponibilidad de backups más recientes, no basarse únicamente en edad.
4. Si falla el watcher o el respaldo externo durante la corrida, el entrenamiento puede continuar solo si aún se mantienen copias recuperables; reparar la protección antes de que la política de retención deje una ventana sin respaldo. Documentar riesgos de perder el volumen remoto entero.
5. No apagar/descartar instancia o almacenamiento hasta tener **checkpoint de M8 final** con SHA verificado en un almacenamiento independiente de Vast.

**Importante:** este kit no incluye un watcher de copias ni un launcher M8. Deben prepararse/probarse en el handoff M8, y son condiciones bloqueantes para dar comienzo a producción. M7.5 prueba el resume y fija aquí su contrato.

## Seguimiento durante M8

- Monitorear train/val loss, LR, grad_norm/clipping, updates omitidos, no finitos, QK logits y clipping, throughput, uso de VRAM, max-load ratio por capa y evolución del mínimo de routing.
- Para MoE, computar las diversidades efectivas de Shannon y Simpson desde los 64 conteos por capa y ventana; revisar evolución, peores capas y sesgos QB. No parar por superar un único umbral de desbalance: exigir persistencia y relación con val loss, gradientes, slips o velocidad.
- **Pausa/investigación inmediata:** NaN/Inf, CUDA OOM repetido, corrupción de checkpoints, error de restore, crecimiento persistente de val loss con señales concurrentes de degradación, o pérdida de cobertura recuperable de backups.
- Validaciones repiten el mismo prefijo de 262.144 tokens; sirven para señal longitudinal pero no estiman generalización sobre muestras independientes. Mantener el split val separado del train.

## Horizonte y wrap del corpus

- Meta: step 183106 = 6.000.017.408 tokens nominales. Quedan 180.664 updates = 5.919.997.952 tokens.
- El corpus train contiene 6.000.001.675 tokens. Después de 183.105 updates completos se habrán consumido 5.999.984.640 tokens; quedan 17.035, insuficientes para otro batch de 32.768 entradas + 1 target.
- `TokenDataLoader` **no** envuelve su propia lectura. Es `train()` quien captura `StopIteration`, llama `reset()` y extrae **un batch completo de 32.768 tokens del prefijo** en el step 183106. El resto de 17.035 tokens se descarta. Esta es la interpretación exacta de la diferencia nominal +15.733, no un batch parcial ni una mezcla de cola y cabecera.

## Coste

El piloto M7.3 observó ~64.194 tok/s end-to-end; a igualdad de tasa, quedarían **~25,6 horas** de sesión entrenadora para los ~5,92B tokens, equivalentes a **~59,4 €** a la tarifa histórica de referencia de 2,32 €/h. Las cadencias de evaluación/checkpoint son distintas y la tasa sostenida, alquiler vigente, exportación, almacenamiento, impuestos y overhead no están certificados. Verificar precio/hora y espacio antes de lanzar. Mantener margen de presupuesto; no suponer una factura exacta.

## Gate de M7.5 a M8

**GO** únicamente tras recibir `m75_resume_receipt.json` con `status=PASS`, analizar el `m75_probe.log`, confirmar que el backup externo permanece verificable, validar espacio/tarifa y tener watcher/backup + launcher probados.

**NO-GO / INVESTIGAR** si se rechaza runtime, source manifest, estado de optimizador, RNG, corpus o cursor; si cambian código/GPU/torch; o si no hay recuperación y almacenamiento asegurados.
