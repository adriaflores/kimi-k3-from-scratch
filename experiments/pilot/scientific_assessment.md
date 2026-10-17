# Pretraining pilot — evaluación científica

**Fecha:** 14 de octubre de 2026  
**Alcance:** 80.019.456 tokens, 2.442 actualizaciones, modelo de 294,7 M parámetros  
**Decisión científica:** los resultados del piloto justificaban continuar hasta aproximadamente 6.000 millones de tokens, sin reinicializar los pesos ni el estado del optimizador.

Este informe documenta los resultados y sus limitaciones, no una secuencia de aprobaciones internas. El piloto es una observación de aprendizaje temprano; por sí solo no establece la calidad ni la generalización del modelo final.

## Aprendizaje y estabilidad

| Señal | Medición | Interpretación |
|---|---:|---|
| Validaciones consecutivas | 20/20 con pérdida decreciente | Aprendizaje consistente sobre un mismo prefijo de validación |
| Pérdida de validación | 7.8979 → 4.1003 | Reducción del 48,1 % respecto a la primera evaluación |
| Pérdida después del warmup | 4.2428 (step 1920) → 4.1003 (step 2442) | Mejora adicional de 0.1425 con LR próximo al pico |
| Gradientes no finitos / actualizaciones omitidas | 0 / 0 | No se detectó inestabilidad numérica en el piloto |
| Actualizaciones con gradient clipping | 146/2442 (5,98 %) | Mayormente concentradas antes del régimen de LR pico |
| QK-Clip | Logit máximo 40,41, umbral 100; 0/78.144 activaciones | Sin clipping de atención en el horizonte observado |

Los resultados apoyaban mantener el LR pico de **3 × 10⁻³** y el tamaño de actualización de **32 × 1024 = 32.768 tokens**. Un segundo piloto con LR de 10⁻³ habría añadido coste, pero no demostraría por sí solo una mejora de generalización. El horizonte completo requiere seguimiento independiente.

## Routing y balance de expertos

| Señal | Medición |
|---|---:|
| Máximo de carga/mean, promedio de doce capas | 1,040× → 2,780× |
| Mayor máximo puntual de carga/mean | 4,715× |
| Expertos con carga cero en las ventanas monitorizadas | 0 en 77 ventanas × 12 capas |
| Diversidad efectiva de Shannon, promedio inicial → final | 63,98 → 62,25 de 64 |
| Diversidad efectiva de Shannon, peor capa final | 60,84 de 64 |
| Diversidad efectiva de Simpson, promedio final | 59,46 de 64 |
| Cuota de los cuatro expertos más elegidos | 11,3 % frente al 6,25 % uniforme |
| Coeficiente de variación de carga final | 0,269 |
| Valor absoluto máximo del bias de Quantile Balancing | 0,822 |

Para cada capa y ventana, con probabilidades empíricas de asignación `p_i = count_i / sum(count)`, se midieron dos diversidades descriptivas: `N_eff = exp(-Σ p_i ln p_i)` (Shannon) y `N_eff,2 = 1 / Σ p_i²` (Simpson). Se promediaron doce capas.

El aumento de la carga del experto más popular indica especialización y cierto desbalance, **no demuestra por sí solo colapso de routing**. En el punto final, la diversidad efectiva media seguía siendo aproximadamente 62,25 de 64 expertos; en las últimas doce ventanas, alrededor de 62,35. Estos recuentos de top-2 no revelan directamente la contribución funcional de cada experto, su calidad por dominio ni el comportamiento del modelo al alcanzar el presupuesto completo.

## Límites de interpretación

- El piloto cubrió solo **1,33 %** del presupuesto total aproximado de entrenamiento.
- Las veinte evaluaciones usaron repetidamente el mismo prefijo de **262.144 tokens**. Permiten observar una tendencia temporal, pero no miden variabilidad entre subconjuntos ni generalización amplia.
- Las métricas MoE recogen asignaciones a expertos, no calidad del especialista ni distribución de cómputo efectivo.
- El piloto terminó en un único proceso. La equivalencia de una reanudación real desde su checkpoint requería una comprobación operativa separada.
- Las curvas y gráficas originales no se distribuyen junto a este informe; las cifras se conservaron del análisis histórico.

## Integridad del checkpoint del piloto

- Actualización final: **2.442**.
- Tokens procesados: **80.019.456**.
- Tamaño del archivo: **2.485.700.423 bytes**.
- SHA-256 del checkpoint: `cd72de269f7036b7a96ffe6f291339e3743eac90a98322d7dea20b9691e17e12`.

La coincidencia de hash del backup fue comunicada tras comprobar el archivo en el equipo local. El archivo original no se incluye en este repositorio y esta edición del informe no constituye una nueva lectura independiente del checkpoint.

## Conclusión

El piloto mostró aprendizaje sostenido, gradientes finitos y un desbalance MoE medido pero no catastrófico. La continuación del pretraining podía conservar el checkpoint, los estados del optimizador, el scheduler, los RNG y los cursores de datos, sin alterar la arquitectura, el tokenizer ni la receta principal. Esta conclusión describe la decisión científica del momento; no equivale a una certificación retrospectiva de todas las métricas del entrenamiento completo.
