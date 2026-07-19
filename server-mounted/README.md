# deepfake-realtime — docker-compose для AlphaFace + seed-vc

Realtime-сервисы на A100-сервере (`master@62.183.4.208`, **SSH порт 22010**),
запускаются через `docker compose` вместо ручных скриптов.

## Было / стало

| Сервис | Порт | Было | Стало |
|---|---|---|---|
| AlphaFace | `127.0.0.1:8001` | `~/launch_alphaface.sh` | `docker compose up -d` |
| seed-vc | `127.0.0.1:17494` | `cd ~/seed-vc && ./venv/bin/python vc_server.py ...` | то же |
| whisper | `127.0.0.1:9000` | отдельный процесс | **выключен, не используется** |

## Управление

```bash
ssh -p 22010 master@62.183.4.208
cd ~/deepfake-realtime

docker compose up -d
docker compose ps
docker compose logs -f alphaface
docker compose restart alphaface
docker compose down
```

`restart: unless-stopped` — сервисы сами поднимутся после перезагрузки сервера,
чего `launch_alphaface.sh` не делал.

## Туннель с ноутбука

```bash
ssh -p 22010 -L 8001:127.0.0.1:8001 -L 17494:127.0.0.1:17494 -N master@62.183.4.208
```

Открывать **http://localhost:8001**. Слева локальный порт, справа удалённый —
`-L 8080:127.0.0.1:8080` уведёт на чужой riskapi-фронтенд, а не на AlphaFace.

После перезапуска контейнеров туннель нужно передёрнуть (процесс за портом
сменился), браузер обновить и заново загрузить source-лицо — WS-сессия рвётся.

## Раскладка на сервере

Оба чекаута перенесены сюда же, `/home/master/deepfake-realtime` (13 ГБ):

```
deepfake-realtime/
├── docker-compose.yml
├── Dockerfile.realtime
├── rt_alphaface_server.py          # сервер AlphaFace
├── rt_alphaface_server.py.premove  # бэкап до правки ALPHA_DIR
├── Alphaface_Official/             # 2.1 ГБ — клон + alphaface_demo.pt
└── seed-vc/                        # 11 ГБ — клон + checkpoints + venv
```

На старых местах оставлены **симлинки**:

```
~/seed-vc            -> ~/deepfake-realtime/seed-vc
~/Alphaface_Official -> ~/deepfake-realtime/Alphaface_Official
~/rt_alphaface_server.py -> ~/deepfake-realtime/rt_alphaface_server.py
```

Они нужны: шебанги внутри venv жёстко прописаны как
`#!/home/master/seed-vc/venv/bin/python3`, и bench-скрипты в `deepface-bridge`
тоже ссылаются на старые пути. Удалите симлинки — сломается `pip` в venv.

Вне этой папки остались (их AlphaFace читает на ходу):
`~/facefusion/.assets/models`, `~/codex_visomaster_test/VisoMaster/model_assets`,
`~/codex_ffhq_realtime_pilot/venv` (venv AlphaFace), `~/.insightface`.

## Почему compose устроен именно так

**Образ пустой, venv'ы монтируются с хоста.** seed-vc требует `torch 2.4.0+cu124`,
AlphaFace — `torch 2.6.0+cu124`. Версии несовместимы, в один образ не собрать.
Плюс ~13 ГБ весов вне git. Поэтому `/home/master` монтируется 1:1 по тому же
абсолютному пути — так отрабатывают и шебанги venv'ов, и пути `FF_MODELS` /
`VM_MODELS` внутри `rt_alphaface_server.py`.

**`network_mode: host` — необходимость.** Все серверы биндят `127.0.0.1` в своём
коде (`rt_alphaface_server.py` — жёстко). На bridge-сети это был бы loopback
*контейнера*, и проброшенные порты вели бы в никуда. Host-сеть оставляет биндинг
там, где его ждут SSH-туннели.

**Ubuntu 24.04 в образе** совпадает с хостом по glibc и Python (3.12.3) — поэтому
смонтированные venv'ы запускаются как есть. Базовый `ubuntu:24.04` не содержит
python3.12, он ставится в `Dockerfile.realtime`.

## Почему whisper выключен

`whisper_server.py` (faster-whisper ASR, порт 9000) не используется ничем:
ни `vc_index.html`, ни `vc_server.py`, ни `rt_alphaface_server.py`, ни клиент
`deepface-bridge/app/app.py` на него не ссылаются. Активных соединений не было,
лог — 121 байт от 17 июля. Он только занимал GPU. Скрипт остался в `seed-vc/`:

```bash
cd ~/deepfake-realtime/seed-vc && ./venv/bin/python whisper_server.py --host 127.0.0.1 --port 9000
```

## Переменные (все опциональны)

| Переменная | Дефолт | Назначение |
|---|---|---|
| `STACK_DIR` | `/home/master/deepfake-realtime` | Где лежат чекауты |
| `MASTER_HOME` | `/home/master` | Корень монтирования |
| `VC_PORT` / `VC_HOST` | `17494` / `127.0.0.1` | seed-vc |
| `SERVICE_UID` / `SERVICE_GID` | `1001` | uid:gid master |

## Эта папка на компе

```
deepfake-realtime/
├── docker-compose.yml      # то, что работает на сервере
├── Dockerfile.realtime     # то, что работает на сервере
├── rt_alphaface_server.py  # копия с сервера, для чтения и правок
├── seed-vc/                # копии vc_server.py, whisper_server.py, vc_index.html
└── legacy/launch_alphaface.sh   # старый способ запуска, больше не нужен
```

Копии кода здесь — только для чтения и правок. Контейнеры запускают оригиналы на
сервере. Поменяли локально — залейте и перезапустите:

```bash
scp -P 22010 rt_alphaface_server.py master@62.183.4.208:/home/master/deepfake-realtime/
ssh -p 22010 master@62.183.4.208 'cd ~/deepfake-realtime && docker compose restart alphaface'
```

Код AlphaFace и seed-vc в git не входит — это upstream-клоны:
`Alphaface_Official` (github.com/andrewyu90/Alphaface_Official) и
`seed-vc` (github.com/Plachtaa/seed-vc).
