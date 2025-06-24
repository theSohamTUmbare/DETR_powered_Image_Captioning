# DETR_powered_image_captioning
The excellent Image captioning model using the DETR inspired architecture

## 🔍 Comparisons 
| Model                       | Image Resolution | BLEU‑4 Score (%) | METEOR Score (%) | Total Time (ms) |
| :-------------------------- | :--------------: | :--------------: | :--------------: | :-------------:  |
| Show & Tell             |     800 × 800    |       27.7       |       23.7       |       210*       |
| Up‑Down                 |     800 × 800    |       36.2       |       27.0       |       620*       |
| M² Transformer          |     800 × 800    |       39.1       |       29.2       |       640*       |
| **DETR‑Powered Captioning** |     **800 × 800**    |       **24.8**       |       **25.7**       |       **200**       |

*runtimes measured according NVIDIA P100, batch=1, identical default decode settings with identical image resultion approximately.
