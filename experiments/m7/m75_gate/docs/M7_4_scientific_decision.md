# Kimi K3 — M7.4 | Dictamen científico del piloto de pretraining

**Fecha:** 14 octubre 2026 · **Run:** `m73_pilot_80m_v1_20261014T065121Z`  
**Decisión: GO a M7.5; aprobar LR pico 3e-3, batch 32×1024, misma receta y estado del checkpoint.**  
**NO autoriza iniciar M8 sin el gate operativo de M7.5.**

## 1. Pregunta de M7.4

Determinar si el piloto desde cero de 80.019.456 tokens proporciona evidencia suficiente de aprendizaje y estabilidad para continuar el pretraining de 6B tokens, o si procede un segundo piloto desde cero a LR 1e-3. Criterio: evidencia global de aprendizaje y riesgo, no umbral arbitrario de loss.

## 2. Resultados y lectura científica

| Señal | Evidencia | Interpretación |
|---|---:|---|
| Evaluaciones | 20/20 reducciones consecutivas | Aprendizaje claro en mismo subconjunto val |
| Val loss | 7.8979 → 4.1003 | −48,1% respecto a primera evaluación |
| Val post-warmup | 4.2428 (step 1920) → 4.1003 (step 2442) | Mejoría 0.1425 con LR ya ≈3e-3 |
| Finite gradients / skipped | 0 no finitos, 0 skips | Sin señal de inestabilidad numérica |
| Gradient clipping | 146/2442 updates, 5,98%; 2 updates registrados tras warmup | Concentrado antes del régimen de LR pico |
| QK-Clip | máximo logit 40,41 ante umbral 100; 0/78.144 clips | Sin saturación de QK en este horizonte |
| Routing experto más cargado, promedio 12 capas | 1,040× → 2.780× | Desigualdad real y creciente |
| Ratio máximo puntual | 4.715× | Episodio localizado que justifica vigilancia |
| Expertos con carga cero | 0 en 77 ventanas × 12 capas | No evidencia de expertos totalmente inactivos a escala de ventana |
| Entropía efectiva media de routing | 63.98 → 62.25 de 64 | Concentración limitada globalmente pese a máximos locales |
| Entropía efectiva peor capa final | 60.84 de 64 | Ninguna capa presenta colapso global de diversidad |
| Entropía de Simpson efectiva media final | 59.46 de 64 | Comprobación alternativa más sensible a concentración |
| Cuota top-4 expertos final | 11.3% vs 6,25% si uniforme | Señal clara de especialización; no de monopolio de rutas |
| Desviación típica relativa carga (CV) final | 0.269 | Desbalance moderado en el agregado |
| Máximo absoluto del QB bias | 0.822 | Control de balance activo; requiere seguimiento |

**Cálculo nuevo (reproducible):** para los 64 `expert_counts` de cada capa y ventana, sea `p_i = count_i / sum(count)`. Diversidad efectiva de Shannon: `N_eff=exp(−Σ p_i ln(p_i))`. Diversidad de Simpson: `N_eff,2=1/(Σ p_i²)`. Se promedian las 12 capas. Es una medida descriptiva de frecuencia de asignación, **no** certifica que los expertos contribuyan funcionalmente por igual, ni que el resultado persistirá hasta 6B tokens. Último punto: ventana parcial de 10 updates, y 327.680 tokens.

### Lectura del routing MoE

El máximo de carga/mean por capa detecta especialistas muy populares, pero **no basta por sí solo para afirmar colapso**. Medido sobre el mismo evento final, la entropía efectiva media equivale a 62.25/64 expertos. Además, la diversidad efectiva media en las últimas 12 ventanas se mantiene alrededor de 62.35 expertos. Con todo, algunas capas y ventanas tienen expertos poco utilizados; no se debe minimizar el riesgo o considerar que QB lo ha resuelto definitivamente.

**Conclusión:** fuerte evidencia de aprendizaje, gradientes estables, ausencia de fallo de attention clipping y concentración MoE cuantificada pero no catastrófica. El LR de 3e-3 no ha mostrado un problema que justifique un piloto comparativo. Un experimento 1e-3 ahora añadiría coste/tiempo y no probaría, por sí solo, que la concentración es dañina ni que el modelo generaliza mejor.

## 3. Gráficos del piloto

![Train y val loss](plots/01_learning_curves.png)

![Grad norm](plots/03_gradient_norms.png)

![Evolución del routing MoE](plots/05_moe_routing_load.png)

![Sesgo QB](plots/08_qb_biases.png)

## 4. Límites de esta aprobación

- El piloto representa solo **1,33%** del horizonte de ~6B tokens. La aprobación no garantiza estabilidad futura.
- Las 20 validaciones repiten el mismo prefijo de 262.144 tokens; permiten medir cambio temporal consistente, no estiman variabilidad entre subconjuntos ni generalización amplia.
- Las métricas de routing resumen **conteos de asignaciones top-2**, no utilización efectiva del cómputo, calidad del especialista o predictividad por dominio.
- El entrenamiento se terminó en un solo proceso desde cero; no se demostró todavía el resume real del checkpoint de M7.3 a YAML de M8.

## 5. Estado de integridad del checkpoint

- Checkpoint final step 2442, 80.019.456 tokens, 2.485.700.423 bytes, SHA-256 `cd72de269f7036b7a96ffe6f291339e3743eac90a98322d7dea20b9691e17e12`.
- El ZIP de M7.3 no contiene pesos y marca el backup externo como pendiente por no incluir `m73_download_receipt.json`.
- **Actualización de evidencia externa al ZIP:** el usuario ejecutó una comprobación en su equipo y mostró el SHA-256 coincidente para `C:/datasets/kimi_k3_backups/m73_pilot_80m_v1_20261014T065121Z_final_last.pt`. Esta salida confirma coincidencia de hash reportada en el PC, aunque el archivo no se ha abierto directamente desde esta sesión y no se ha aportado recibo JSON. No es necesario repetir la descarga solo para generar un archivo de recibo, salvo que se requiera por la política final de M7.5.

## 6. Cierre

**M7.4 — GO científico.** Recomendación: no lanzar segundo piloto, no modificar LR, arquitectura, batch, tokenizer ni código. Proceder a M7.5 para probar la compatibilidad del resume real, blindar el backup y cerrar operación de M8. Una vez M7.5 sea PASS, continuar desde `step=2442` hasta `max_steps=183106` conservando el optimizador, scheduler, RNG y cursores.
