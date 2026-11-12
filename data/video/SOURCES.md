# Fontes dos clipes de vídeo

Licença e URL de cada clipe usado pelo Analisador de Vídeo (ESTRATEGIA.md, seção 4).
Os clipes em si não vão para o git; `scripts/download_data.py video` os reproduz a partir de
`data/video/clipes.json`, que também guarda o gabarito (Desvios e Quedas esperados) de cada um.

| Arquivo | Cenário | Fonte (URL) | Autor | Licença |
| --- | --- | --- | --- | --- |
| `Squat_and_Frontal_Raise.webm` | Sessão de Fisioterapia: agachamento, vista lateral | [Wikimedia Commons](https://commons.wikimedia.org/wiki/File:Squat_and_Frontal_Raise.webm) | Taco fleur | CC BY-SA 4.0 |
| `Squat_-_exercise_demonstration_video.webm` | Sessão de Fisioterapia: agachamento, vista de costas | [Wikimedia Commons](https://commons.wikimedia.org/wiki/File:Squat_-_exercise_demonstration_video.webm) | FitnessScape | CC BY 3.0 |
| `unipodal_S1.ogv` | Sessão de Fisioterapia: agachamento unipodal, joelho alinhado | [Wikimedia Commons](https://commons.wikimedia.org/wiki/File:Validity-and-inter-rater-reliability-of-medio-lateral-knee-motion-observed-during-a-single-limb-1471-2474-11-265-S1.ogv) | Ageberg et al., *BMC Musculoskeletal Disorders* 11:265 (2010) | CC BY 2.0 |
| `unipodal_S2.ogv` | Sessão de Fisioterapia: agachamento unipodal, joelho em valgo | [Wikimedia Commons](https://commons.wikimedia.org/wiki/File:Validity-and-inter-rater-reliability-of-medio-lateral-knee-motion-observed-during-a-single-limb-1471-2474-11-265-S2.ogv) | Ageberg et al., *BMC Musculoskeletal Disorders* 11:265 (2010) | CC BY 2.0 |
| `fall-01-cam0.mp4` … `fall-05-cam0.mp4` | Quarto: queda (5 sequências) | [UR Fall Detection Dataset](https://fenix.ur.edu.pl/~mkepski/ds/uf.html) | Kwolek & Kepski, *Comput. Methods Programs Biomed.* 117(3) (2014) | CC BY-NC-SA 4.0 (uso acadêmico) |
| `adl-01-cam0.mp4` … `adl-03-cam0.mp4` | Quarto: atividades de vida diária, sem queda | [UR Fall Detection Dataset](https://fenix.ur.edu.pl/~mkepski/ds/uf.html) | Kwolek & Kepski (2014) | CC BY-NC-SA 4.0 (uso acadêmico) |

Os vídeos do UR Fall juntam, lado a lado, o mapa de profundidade (metade esquerda) e a imagem RGB
(metade direita) do Kinect; o Analisador usa só a metade RGB (`recorte` no `clipes.json`).

Pesos do modelo: `yolov8n-pose.pt` (Ultralytics, AGPL-3.0), baixado automaticamente para
`data/video/` na primeira análise.
