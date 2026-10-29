# Nemotron 3 Diarization · MAX CPU

Два HTTP-эндпоинта для `nvidia/Nemotron-3-Diarization`. Модель компилируется
MAX 26.7.0 при запуске и работает на CPU в float32. Каждый запрос независим.

```text
float32 audio → log-mel → MAX → float32 logits
float32 mel features    → MAX → float32 logits
```

## Запуск

Python 3.12, Linux x86-64. В локальном каталоге checkpoint должны находиться
`config.json`, `processor_config.json` и `model.safetensors` либо shards с индексом.
Веса NVIDIA распространяются отдельно по OpenMDW 1.1.

```bash
# Для окружений с недоступным для записи ~/.cache:
export XDG_CACHE_HOME="$PWD/.cache"
export UV_CACHE_DIR="$PWD/.cache/uv"

uv venv
source .venv/bin/activate
uv pip install -e .
nemotron-max --model models/nemotron --host 0.0.0.0 --port 8000
```

HTTP-слой — Starlette/Uvicorn с двумя маршрутами над MAX Engine. FastAPI и
Pydantic в обработке запросов не используются. `max serve` не используется.
CLI только запускает сервер. Старые команды `serve`, `download`, `diarize` удалены.

## API

Тело запроса и ответа — `application/octet-stream`, little-endian `float32`,
без заголовка файла, JSON, multipart или base64.

| Endpoint | Вход |
|---|---|
| `POST /infer/audio` | Mono waveform, 16 kHz, не менее 160 сэмплов. WAV/MP3 не принимаются. |
| `POST /infer/features` | Log-mel матрица `[T, 128]` в row-major порядке. Ожидаются совместимые признаки с шагом 10 ms. |

Ответ — **сырые логиты** `[T, 8]`, спикеры в порядке первого появления.
Заголовки `X-Shape: T,8`, `X-Dtype: float32-le`, `X-Frame-Seconds: 0.01`
описывают матрицу. Для аудио последний служебный кадр frontend исключается.
Для признаков все переданные кадры считаются действительными.

```python
import httpx
import numpy as np

audio = np.asarray(mono_audio_16khz, dtype="<f4")
with httpx.Client(base_url="http://localhost:8000", timeout=300) as client:
    response = client.post(
        "/infer/audio",
        content=audio.tobytes(),
        headers={"Content-Type": "application/octet-stream"},
    )
    response.raise_for_status()
    shape = tuple(map(int, response.headers["X-Shape"].split(",")))
    logits = np.frombuffer(response.content, dtype="<f4").reshape(shape)
    active_speakers = logits > 0  # эквивалент вероятности > 0.5
```

Для готовых признаков отправьте `mel_features.astype("<f4").tobytes()` на
`/infer/features`. Последовательные запросы через один HTTP-клиент переиспользуют
соединение.

## Выполнение

В сервисе нет сессий, TTL, кэша результатов, авторизации, декодирования файлов,
ресэмплинга, построения интервалов спикеров или служебных endpoints.
`/healthz`, `/docs`, `/openapi.json` и старые `/v1/...` удалены.
Числа читаются через `np.frombuffer`; ответ не преобразуется в Python-списки.
На обычном пути не вычисляется даже sigmoid: клиент получает логиты модели.

Один процесс держит одну модель. CPU-инференс выполняется последовательно;
ожидающие запросы не занимают заблокированные worker threads. MAX сам использует
потоки CPU. Автоматического batching и гарантии real-time нет.

**Speaker cache/FIFO остаются внутри одного запроса:** это часть алгоритма модели
для обработки длинной записи. Между запросами они не сохраняются. Поэтому
спикер `0` в одном запросе может соответствовать другому человеку в следующем.
Для согласованных меток передавайте запись целиком одним запросом.

## Проверки

```bash
uv pip install torch --index-url https://download.pytorch.org/whl/cpu
uv pip install -e '.[test,reference]'
pytest -q

python scripts/validate_reference.py --model models/nemotron \
  --stream-steps 90 --output validation-trained.json
```

Эталон Transformers закреплён на commit
`420dca92a4cf7d96dfbc1cc46f00375b0681ff8f`. PyTorch/NeMo не нужны серверу.
Численные проверки и HTTP-тесты используют воспроизводимые тестовые веса.
Полная 31-слойная конфигурация проверена на CPU на 90 чанках; исторический отчёт
до упрощения HTTP API — `docs/validation-cpu-synthetic.json`.
Настоящие веса, качество на речи и GPU пока не проверены.
Подробности модели: [архитектура](docs/architecture.md).
