# Experiments:
### 8th Sept ----- multimodal vs text-only prompt learning within 500 samples (Fakeddit)
#### configs: 
`epoch = 5, clip_model = ViT-B/32, prompt_model = roberta-base, len(text)>10`
#### 2-shot:
| models       | f1-fake | f1-real | macro-f1| acc |
| ----------- | -----------  | -----------  | ----------- |----------- |
| text-only   |0,0.30,0.16 | 0.71,0.72,0.69  |0.35,0.51,0.42  | 0.54,0.60,0.54|
| multi      |0.06,0.24,0.52        |0.71,0.73,0.67      |0.39,0.49,0.60 |0.56,0.60,0.61   |
| text-only  |0.15±0.12        |0.70±0.01    |0.42±0.07 |0.56±0.03   |
| multi      | **0.27**±0.19        |0.70±0.02      |**0.49**±0.08 |**0.59**±0.02   |
#### 4-shot:
| models       | f1-fake | f1-real | macro-f1| acc |
| ----------- | -----------  | -----------  | ----------- |----------- |
| text-only   |0.62,0,0.65 | 0.08,0.71,0.32  |0.35,0.35,0.49  | 0.45,0.55,0.53|
| multi      |0.09,0.59,0.52        |0.71,0.75,0.53      |0.40,0.67,0.52 |0.56,0.68,0.52   |
| text-only  |**0.42**±0.29        |0.37±0.25    |0.39±0.06 |0.51±0.04   |
| multi      | 0.4±0.22        |**0.66**±0.09      |**0.53**±0.11 |**0.59**±0.06   |
#### 100-shot:
| models       | f1-fake | f1-real | macro-f1| acc |
| ----------- | -----------  | -----------  | ----------- |----------- |
| text-only   |0.68,0.47,0.62 | 0.79,0.78,0.77  |0.74,0.62,0.70  | 0.75,0.68,0.71|
| multi      |0.68,0.64,0.65        |0.74,0.56,0.55      |0.71,0.60,0.60 |0.71,0.60,0.61   |
| text-only  |0.59±0.08       |**0.78**±0.01    |**0.68**±0.04 |**0.71**±0.02   |
| multi      | **0.66**±0.25        |0.62±0.03      |0.63±0.05 |0.64±0.04  |

### 15th Sept ----- multimodal with different alpha and 10 random seeds on Politifact

#### 2-shot: `epoch=20, alpha=0~1, seed=1`

| models       | f1-fake | f1-real | macro-f1| acc |
| ----------- | -----------  | -----------  | ----------- |----------- |
| multi      |0.20,0.65,0.59,0.65,0,0.65,0.65,0.64,0.62,0.53,0.62       |0.60,0,0.3,0,0.65,0,0,0.04,0,0.09,0    |0.40,0.33,0.44,0.33,0.36,0.32,0.32,0.34,0.31,0.31,0.31 |0.46,0.48,0.48,0.48,0.49,0.48,0.48,0.47,0.45,0.38,0.45  |

#### 2-shot: `epoch=20, alpha=0~1, seed=2`

| models       | f1-fake | f1-real | macro-f1| acc |
| ----------- | -----------  | -----------  | ----------- |----------- |
| multi      |0.63,0.33,0.71,0.34,0,0.41,0.17,0.38,0,0.08,0.38     |0,0.66,0.68,0.73,0.68,0.71,0.70,0.74,0.68,0.69,0.72   |0.32,0.49,0.70,0.53,0.34,0.56,0.44,0.56,0.34,0.39,0.55 |0.46,0.55,0.70,0.61,0.52,0.61,0.56,0.63,0.52,0.54,0.62 |

#### 2-shot: `epoch=20, alpha=0~1, seed=3`
| models       | f1-fake | f1-real | macro-f1| acc |
| ----------- | -----------  | -----------  | ----------- |----------- |
| multi      |0.58,0.38,0.62,0.64,0.37,0.24,0.14,0.60,0.66,0.47,0.67      |0.45,0.40,0.69,0.64,0.71,0.67,0.69,0.72,0.37,0.67,0.34    |0.52,0.39,0.66,0.64,0.54,0.45,0.41,0.66,0.51,0.54,0.50|0.53,0.39,0.66,0.64,0.60,0.54,0.54,0.68,0.56,0.58,0.56 |

#### 2-shot: `epoch=20, alpha=0~1, seed=4`

| models       | f1-fake | f1-real | macro-f1| acc |
| ----------- | -----------  | -----------  | ----------- |----------- |
| multi      |0.62,0.66,0.50,0.08,0.47,0.30,0.42,0.08,0.39,0.08,0.14      |0.29,0.04,0.60,0.69,0.73,0.72,0.73,0.69,0.73,0.69,0.69    |0.46,0.35,0.55,0.39,0.60,0.51,0.58,0.39,0.56,0.39,0.41 |0.51,0.49,0.55,0.54,0.64,0.60,0.63,0.53,0.62,0.54,0.55 |

#### 2-shot: `epoch=20, alpha=0~1, seed=5`

| models       | f1-fake | f1-real | macro-f1| acc |
| ----------- | -----------  | -----------  | ----------- |----------- |
| multi      |0.57,0.08,0.46,0.23,0.37,0.30,0.54,0.45,0.45,0.36,0.36      |0.54,0.68,0.69,0.71,0.72,0.72,0.72,0.73,0.73,0.71,0.71    |0.56,0.38,0.58,0.47,0.55,0.51,0.63,0.59,0.59,0.54,0.54 |0.56,0.52,0.61,0.58,0.61,0.60,0.65,0.64,0.64,0.60,0.60 |

#### 4-shot: `epoch=20, alpha=0~1, seed=1`

| models       | f1-fake | f1-real | macro-f1| acc |
| ----------- | -----------  | -----------  | ----------- |----------- |
| multi      |0.00,0.04,0.63,0.12,0.64,0.58,0.69,0.74,0.12,0.48,0.69|0.67,0.67,0.16,0.68,0.76,0.71,0.78,0.80,0.69,0.74,0.79|0.33,0.36,0.39,0.40,0.70,0.65,0.73,0.77,0.41,0.61,0.74 |0.50,0.51,0.48,0.53,0.72,0.66,0.74,0.77,0.54,0.65,0.75 |

#### 4-shot: `epoch=20, alpha=0~1, seed=2`

| models       | f1-fake | f1-real | macro-f1| acc |
| ----------- | -----------  | -----------  | ----------- |----------- |
| multi      |0.66,0.45,0.65,0.00,0.65,0.74,0.32,0.33,0.75,0.06,0.57|0.28,0.46,0.18,0.68,0.13,0.73,0.71,0.71,0.65,0.69,0.63|0.47,0.45,0.41,0.34,0.39,0.73,0.52,0.52,0.70,0.38,0.60 | 0.54,0.45,0.51,0.52,0.50,0.73,0.59,0.59,0.71,0.53,0.60 |

#### 4-shot: `epoch=20, alpha=0~1, seed=3`

| models       | f1-fake | f1-real | macro-f1| acc |
| ----------- | -----------  | -----------  | ----------- |----------- |
| multi    |0.25,0.31,0.24,0.50,0.57,0.71,0.72,0.55,0.06,0.47,0.69|0.71,0.51,0.63,0.71,0.62,0.80,0.79,0.74,0.69,0.75,0.79|0.48,0.41,0.44,0.60,0.59,0.76,0.76,0.64,0.38,0.61,0.74|0.58,0.43,0.51,0.63,0.59,0.76,0.75,0.67,0.53,0.66,0.75|

#### 4-shot: `epoch=20, alpha=0~1, seed=4`

| models       | f1-fake | f1-real | macro-f1| acc |
| ----------- | -----------  | -----------  | ----------- |----------- |
| multi      |0.78,0.00,0.22,0.36,0.48,0.37,0.38,0.40,0.51,0.41,0.39|0.84,0.68,0.69,0.71,0.73,0.71,0.72,0.73,0.73,0.73,0.73|0.81,0.34,0.46,0.54,0.60,0.54,0.55,0.57,0.62,0.57,0.56|0.82,0.52,0.56,0.61,0.64,0.61,0.62,0.63,0.65,0.63,0.62|


#### 4-shot: `epoch=20, alpha=0~1, seed=5`

| models       | f1-fake | f1-real | macro-f1| acc |
| ----------- | -----------  | -----------  | ----------- |----------- |
| multi      |0.70,0.14,0.58,0.67,0.37,0.53,0.55,0.53,0.62,0.36,0.77|0.62,0.69,0.56,0.48,0.72,0.74,0.76,0.74,0.69,0.72,0.67|0.66,0.42,0.57,0.57,0.55,0.63,0.65,0.64,0.65,0.54,0.72|0.67,0.55,0.57,0.59,0.61,0.66,0.68,0.67,0.66,0.61,0.73|


#### 8-shot: `epoch=20, alpha=0~1, seed=1`

| models       | f1-fake | f1-real | macro-f1| acc |
| ----------- | -----------  | -----------  | ----------- |----------- |
| multi      |0.27,0.65,0.38,0.69,0.61,0.73,0.57,0.75,0.81,0.61,0.58|0.67,0.00,0.73,0.72,0.65,0.46,0.77,0.54,0.80,0.76,0.77|0.47,0.33,0.56,0.70,0.63,0.59,0.67,0.65,0,80,0.68,0.68|0.54,0.48,0.63,0.70,0.63,0.64,0.70,0.68,0.80,0.70,0.70|


#### 8-shot: `epoch=20, alpha=0~1, seed=2`

| models       | f1-fake | f1-real | macro-f1| acc |
| ----------- | -----------  | -----------  | ----------- |----------- |
| multi      |0.50,0.64,0.66,0.65,0.57,0.76,0.48,0.79,0.76,0.76,0.73|0.54,0.63,0.19,0.02,0.77,0.77,0.75,0.77,0.74,0.60,0.82|0.52,0.63,0.42,0.34,0.67,0.76,0.62,0.78,0.75,0.68,0.77|0.52,0.63,0.52,0.49,0.70,0.76,0.66,0.78,0.75,0.70,0.78|


#### 8-shot: `epoch=20, alpha=0~1, seed=3`

| models       | f1-fake | f1-real | macro-f1| acc |
| ----------- | -----------  | -----------  | ----------- |----------- |
| multi      |0.12,0.46,0.43,0.83,0.68,0.79,0.66,0.67,0.84,0.82,0.69|0.69,0.53,0.72,0.79,0.24,0.79,0.04,0.17,0.84,0.76,0.30|0.41,0.50,0.58,0.81,0.46,0.79,0.35,0.42,0.84,0.79,0.50|0.54,0.50,0.63,0.81,0.55,0.79,0.49,0.53,0.84,0.80,0.57|


#### 8-shot: `epoch=20, alpha=0~1, seed=4`

| models       | f1-fake | f1-real | macro-f1| acc |
| ----------- | -----------  | -----------  | ----------- |----------- |
| multi      |0.00,0.65,0.40,0.44,0.44,0.67,0.62,0.61,0.65,0.52,0.46|0.68,0.54,0.74,0.71,0.74,0.19,0.76,0.75,0.70,0.72,0.72|0.34,0.60,0.57,0.58,0.59,0.43,0.69,0.68,0.67,0.62,0.59|0.52,0.60,0.64,0.62,0.64,0.53,0.71,0.70,0.68,0.65,0.63|


#### 8-shot: `epoch=20, alpha=0~1, seed=5`

| models       | f1-fake | f1-real | macro-f1| acc |
| ----------- | -----------  | -----------  | ----------- |----------- |
| multi      |0.10,0.65,0.52,0.38,0.63,0.78,0.80,0.61,0.70,0.73,0.79|0.66,0.02,0.72,0.73,0.77,0.83,0.76,0,78,0.80,0.61,0.83|0.38,0.34,0.62,0.55,0.70,0.80,0.78,0.69,0.75,0.67,0.81|0.51,0.48,0.65,0.62,0.71,0.81,0.79,0.71,0.76,0.68,0.81|

#### 16-shot: `epoch=20, alpha=0~1, seed=1`

| models       | f1-fake | f1-real | macro-f1| acc |
| ----------- | -----------  | -----------  | ----------- |----------- |
| multi      |0.56,0.65,0.64,0.28,0.51,0.63,0.31,0.75,0.33,0.76,0.36|0.26,0.18,0.04,0.71,0.74,0.68,0.71,0.80,0.72,0.66,0.72|0.41,0.42,0.34,0.50,0.63,0.66,0.51,0.77,0.52,0.71,0.54|0.45,0.51,0.48,0.59,0.66,0.66,0.60,0.78,0.60,0.72,0.61|



#### 16-shot: `epoch=20, alpha=0~1, seed=2`

| models       | f1-fake | f1-real | macro-f1| acc |
| ----------- | -----------  | -----------  | ----------- |----------- |
| multi      |0.62,.52,0.66,0.78,0.65,0.79,0.78,0.38,0.72,0.43,0.80|0.06,0.58,0.55,0.74,0.77,0.84,0.67,0.74,0.81,0.74,0.83|0.34,0.55,0.61,0.76,0.71,0.82,0.72,0.56,0.76,0.58,0.81|0.46,0.55,0.61,0.77,0.72,0.82,0.73,0.63,0.77,0.64,0.81|

#### 16-shot: `epoch=20, alpha=0~1, seed=3`

| models       | f1-fake | f1-real | macro-f1| acc |
| ----------- | -----------  | -----------  | ----------- |----------- |
| multi      |0.35,0.71,0.70,0.71,0.77,0.81,0.66,0.82,0.76,0.80,0.86|0.52,0.50,0.73,0.81,0.75,0.82,0.77,0.77,0.82,0.80,0.86|0.44,0.63,0.72,0.75,0.76,0.82,0.71,0.79,0.79,0.80,0.86|0.45,0.61,0.72,0.76,0.76,0.82,0.72,0.80,0.80,0.80,0.86|

#### 16-shot: `epoch=20, alpha=0~1, seed=4`

| models       | f1-fake | f1-real | macro-f1| acc |
| ----------- | -----------  | -----------  | ----------- |----------- |
| multi      |0.14,0.62,0.74,0.40,0.73,0.78,0.69,0.73,0.81,0.79,0.78|0.65,0.53,0.80,0.74,0.78,0.81,0.36,0.79,0.82,0,73,0.83|0.40,0.57,0.77,0.57,0.76,0.79,0.52,0.76,0.81,0.76,0.80|0.50,0.58,0.77,0.63,0.76,0.80,0.58,0.77,0.81,0.77,0.81|



#### 16-shot: `epoch=20, alpha=0~1, seed=5`

| models       | f1-fake | f1-real | macro-f1| acc |
| ----------- | -----------  | -----------  | ----------- |----------- |
| multi      |0.00,0.65,0.65,0.72,0.43,0.76,0.81,0.80,0.87,0.78,0.82|0.67,0.16,0.02,0.60,0.73,0.83,0.83,0.82,0.89,0.83,0.80|0.34,0.41,0.34,0.66,0.58,0.80,0.82,0.81,0.88,0.80,0.81|0.51,0.51,0.49,0.67,0.64,0.80,0.82,0.81,0.88,0.81,0.81|


#### 100-shot: `epoch=20, alpha=0~1, seed=1`

| models       | f1-fake | f1-real | macro-f1| acc |
| ----------- | -----------  | -----------  | ----------- |----------- |
| multi      |0.00,0.58,0.47,0.84,0.78,0.87,0.90,0.80,0.79,0.80,0.87|0.69,0.78,0.76,0.88,0.83,0.90,0.92,0.83,0.82,0.85,0.89|0.35,0.68,0.62,0.86,0.80,0.88,0.91,0.82,0.81,0.83,0.88|0.53,0.71,0.67,0.86,0.81,0.88,0.91,0.82,0.81,0.83,0.88|

#### 100-shot: `epoch=20, alpha=0~1, seed=2`

| models       | f1-fake | f1-real | macro-f1| acc |
| ----------- | -----------  | -----------  | ----------- |----------- |
| multi      |0.09,0.78,0.85,0.43,0.89,0.85,0.90,0.90,0.90,0.89,0.88|0.70,0.82,0.89,0.73,0.90,0.85,0.91,0.91,0.91,0.90,0.89|0.40,0.80,0.87,0.58,0.89,0.85,0.90,0.90,0.90,0.89,0.88|0.55,0.80,0.87,0.64,0.89,0.85,0.90,0.90,0.90,0.89,0.88|


#### 100-shot: `epoch=20, alpha=0~1, seed=3`

| models       | f1-fake | f1-real | macro-f1| acc |
| ----------- | -----------  | -----------  | ----------- |----------- |
| multi      |0.31,0.38,0.72,0.87,0.86,0.87,0.80,0.86,0.89,0.80,0.86|0.43,0.72,0.80,0.9,0.87,0.90,0.83,0.88,0.90,0.83,0.88|0.37,0.55,0.76,0.88,0.86,0.88,0.82,0.87,0.89,0.82,0.87|0.37,0.62,0.77,0.88,0.86,0.88,0.82,0.87,0.89,0.82,0.87|



#### 100-shot: `epoch=20, alpha=0~1, seed=4`

| models       | f1-fake | f1-real | macro-f1| acc |
| ----------- | -----------  | -----------  | ----------- |----------- |
| multi      |0.12,0.62,0.86,0.81,0.77,0.86,0.81,0.85,0.83,0.88,0.85|0.70,0.64,0.84,0.76,0.82,0.87,0.82,0.85,0.85,0.89,0.87|0.41,0.63,0.85,0.78,0.79,0.86,0.82,0.85,0.84,0.88,0.86|0.55,0.63,0.85,0.79,0.80,0.86,0.82,0.85,0.84,0.88,0.86|


#### 100-shot: `epoch=20, alpha=0~1, seed=5`

| models       | f1-fake | f1-real | macro-f1| acc |
| ----------- | -----------  | -----------  | ----------- |----------- |
| multi      |0.21,0.54,0.66,0.90,0.88,0.83,0.91,0.88,0.86,0.80,0.88|0.67,0.73,0.15,0.91,0.88,0.87,0.92,0.88,0.87,0.85,0.89|0.44,0.64,0.40,0.90,0.88,0.85,0.91,0.88,0.86,0.83,0.88|0.53,0.66,0.51,0.90,0.88,0.85,0.91,0.88,0.86,0.83,0.88|
