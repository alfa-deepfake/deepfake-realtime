# deepfake-realtime

AlphaFace (face swap) + seed-vc (voice conversion) под docker compose.

---

## Портируемый вариант — для нового компа

```bash
docker compose up -d --build
```

Больше ничего. Образы сами соберут окружение, склонируют код моделей и скачают
все веса при первом запуске.

- AlphaFace → **http://localhost:8001**
- seed-vc → **http://localhost:17494**

**Требования к машине:** Linux, NVIDIA-драйвер + `nvidia-container-toolkit`
(проверить: `docker info | grep -i nvidia`), ~10 ГБ VRAM на оба сервиса,
Docker Compose v2+, ~35 ГБ свободного диска.

**Первый запуск долгий:** ~20 ГБ образов собирается 15–20 минут, плюс ~5.6 ГБ
весов качается при старте. Последующие запуски — секунды, всё лежит в volume'ах.

Что качается само:

| Сервис | Что | Откуда | Размер |
|---|---|---|---|
| alphaface | `alphaface_demo.pt`, `arcface_w600k_r50` | Google Drive | 2.0 ГБ |
| alphaface | bisenet, xseg, gfpgan, GPEN-BFR-512, occluder | facefusion / visomaster releases | 807 МБ |
| alphaface | insightface `buffalo_l` | insightface releases | 600 МБ |
| seed-vc | Seed-VC, CosyVoice, campplus, wav2vec2, bigvgan | HuggingFace | 2.6 ГБ |

Google Drive иногда режет скорость и отдаёт 503 — скачивание переживает это
ретраями. Если совсем не идёт, положите файлы руками:

```bash
docker compose cp alphaface_demo.pt alphaface:/models/alphaface_demo.pt
docker compose restart alphaface
```

### Настройка

Все переменные опциональны:

| Переменная | Дефолт | Назначение |
|---|---|---|
| `ALPHAFACE_PORT` | `8001` | Порт AlphaFace на хосте |
| `SEEDVC_PORT` | `17494` | Порт seed-vc на хосте |
| `BIND_ADDR` | `127.0.0.1` | На чём слушать. **Не ставьте `0.0.0.0`** — у сервисов нет авторизации |
| `ALPHAFACE_DEMO_GDRIVE_ID` | id весов | Если Drive заблокировал дефолтный файл |

Доступ с другой машины — только через SSH-туннель:

```bash
ssh -L 8001:127.0.0.1:8001 -L 17494:127.0.0.1:17494 -N user@host
```

### Структура

```
docker-compose.yml           # портируемый стек
alphaface/
├── Dockerfile               # torch 2.6.0+cu124 + insightface + onnxruntime-gpu
├── entrypoint.sh            # качает веса в volume, потом запускает сервер
├── requirements.txt
└── rt_alphaface_server.py   # правленый: пути и биндинг через env
seed-vc/
├── Dockerfile               # torch 2.4.0+cu124, клон seed-vc @51383ef
├── entrypoint.sh
├── requirements.txt
├── vc_server.py
└── vc_index.html
server-mounted/              # вариант, работающий сейчас на A100
legacy/                      # launch_alphaface.sh, whisper_server.py — не нужны
```

### Почему два образа, а не один

seed-vc требует `torch 2.4.0+cu124`, AlphaFace — `torch 2.6.0+cu124`. Версии
несовместимы, в одно окружение не ставятся.

Ни один образ не основан на CUDA-образе: pinned-колёса torch и onnxruntime-gpu несут
свои cu124-библиотеки, а системная CUDA поверх них как раз и давала
`CUDNN_STATUS_SUBLIBRARY_VERSION_MISMATCH`. Работает связка «чистая Ubuntu +
nvidia-container-toolkit (только драйвер)».

### Грабли, на которые уже наступили

Всё это уже учтено в файлах — не «почините» обратно:

- **`opencv-python-headless` строго 4.10.x.** С 4.12 нужен numpy≥2, а insightface
  и pinned torch собраны под numpy 1.x — install не разрешается.
- **`onnxruntime` намеренно отсутствует в seed-vc.** Он требует protobuf≥4.25,
  а `descript-audiotools` — protobuf<3.20. seed-vc импортирует onnxruntime
  только лениво, в неиспользуемом RMVPE-onnx пути.
- **insightface собирается из исходников** — для Python 3.12 готового wheel нет,
  поэтому в образ временно ставится `build-essential` (и удаляется в том же слое).
- **insightface игнорирует `INSIGHTFACE_HOME`** и лезет в `~/.insightface`.
  Entrypoint симлинкает его в volume, иначе 600 МБ качаются заново при каждом
  пересоздании контейнера.
- **`gdown` 6.x убрал флаг `--id`** — id передаётся позиционно.

---

## Серверный вариант — то, что работает сейчас

Лежит в `server-mounted/`, на сервере — в `/home/master/deepfake-realtime`.
Он **не** самодостаточен: монтирует venv'ы и веса с хоста и привязан к путям
именно этой машины. Подробности — в `server-mounted/README.md`.

```bash
ssh -p 22010 master@0.0.0.0
cd ~/deepfake-realtime
docker compose ps
```

Порты те же — 8001 и 17494. Туннель:

```bash
ssh -p 22010 -L 8001:127.0.0.1:8001 -L 17494:127.0.0.1:17494 -N master@62.183.4.208
```

Whisper (порт 9000) выключен: на него не ссылался ни один компонент.
