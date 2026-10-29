# Nemotron 3 Diarization на Modular MAX

Реализация `nvidia/Nemotron-3-Diarization` для **MAX 26.7.0, CPU, float32**.
Вычисления feature stacking, Transformer, RoPE, upsampler и классификатора
выполняются скомпилированными MAX-графами. Управление чанками и AOSC/FIFO — NumPy.
Log-mel preprocessing — NumPy/SciPy. PyTorch и NeMo для работы сервиса не нужны;
PyTorch используется только для сравнения с эталоном.

Это отдельный HTTP-сервис над MAX Engine. Он не регистрирует модель в штатном
`max serve`: его LLM scheduler и chat-completions API не реализуют диаризацию.

## Установка и файлы модели

Python 3.12, Linux x86-64. Отладка выполнялась на CPU; GPU не проверен.

```bash
# В окружениях с read-only ~/.cache, включая текущее окружение Codex:
export XDG_CACHE_HOME="$PWD/.cache"
export UV_CACHE_DIR="$PWD/.cache/uv"

uv venv
source .venv/bin/activate
uv pip install -e .
```

В `models/nemotron/` нужны:

```text
config.json
processor_config.json
model.safetensors
```

Поддерживаются также shards с `model.safetensors.index.json`. Загружаются только
веса safetensors; код из репозитория модели не исполняется. Архивы `.nemo` и GGUF
напрямую не читаются. Лицензия весов — OpenMDW 1.1, см. карточку NVIDIA.

Можно скачать checkpoint с закреплением commit ID:

```bash
nemotron-max download --revision <MODEL_COMMIT_SHA> --output models/nemotron
```

В текущем окружении предоставленные JSON уже сохранены в `models/nemotron/`.
Исходный файл весов из вложения не перенесён: загрузчик вложений имеет лимит
32 MiB. Прямое скачивание с Hugging Face также заблокировано сетевой политикой.
Эти ограничения относятся к доступу к весам в окружении, а не к MAX.

Для передачи небольшими файлами выполните на машине с checkpoint:

```bash
python scripts/shard_checkpoint.py /path/to/checkpoint /path/to/sharded --max-mib 24
```

Передайте все получившиеся `.safetensors`, индекс и оба JSON. Разбиение не меняет
значения тензоров. Сам скрипт требует только `safetensors` и `numpy`.

## HTTP API для внешнего приложения

```bash
export DIARIZATION_API_KEY='replace-with-your-key'
nemotron-max serve --model models/nemotron --host 0.0.0.0 --port 8000
```

По умолчанию CLI слушает `127.0.0.1`. Если `DIARIZATION_API_KEY` задан,
рабочие endpoints требуют `Authorization: Bearer ...`. `/healthz` открыт.
Для внешнего доступа настройте HTTPS на своём reverse proxy; ограничьте там
размер HTTP body. Сам сервис проверяет размер загруженного файла перед
декодированием и длительность аудио по заголовку перед чтением сэмплов.

Компиляция происходит при запуске. `/healthz` доступен после загрузки модели.
Один процесс обслуживает одну копию модели; вызовы MAX сериализованы.

```bash
curl http://localhost:8000/v1/audio/diarizations \
  -H "Authorization: Bearer $DIARIZATION_API_KEY" \
  -F file=@meeting.wav
```

Пример структуры ответа:

```json
{
  "duration": 3.0,
  "frame_seconds": 0.01,
  "segments": [
    {"speaker": 0, "start": 0.12, "end": 1.54},
    {"speaker": 1, "start": 1.2, "end": 2.97}
  ]
}
```

Это иллюстрация формата, не измеренный результат. Перекрывающиеся интервалы
допускаются. Спикеры нумеруются с нуля в порядке первого появления.
Модель выдаёт метки спикеров и время; для текста нужен отдельный ASR.

Поддерживаются аудиоформаты, декодируемые установленным libsndfile, в частности
WAV/FLAC. Mono/stereo, 8–192 kHz; сервис приводит к mono и 16 kHz.
По умолчанию максимум 64 MiB на файл и 600 секунд аудио.
Документация API: `/docs`, схема: `/openapi.json`.

Python-клиент:

```python
import requests

with open("meeting.wav", "rb") as audio:
    response = requests.post(
        "http://localhost:8000/v1/audio/diarizations",
        headers={"Authorization": "Bearer YOUR_KEY"},
        files={"file": audio},
        timeout=300,
    )
response.raise_for_status()
print(response.json()["segments"])
```

## Потоковый интерфейс модели

`POST /v1/diarization/sessions` создаёт независимое состояние. Далее:

```text
POST /v1/diarization/sessions/{session_id}
{
  "input_features": [[128 float values per frame], ...],
  "num_lookahead_frames": 4,
  "num_lookback_frames": 0,
  "is_last": false
}
```

`input_features` — **log-mel признаки, не PCM**. Это интерфейс для приложений,
которые уже имеют совместимый frontend. Все кадры mel имеют шаг 10 ms;
`num_lookahead_frames`/`num_lookback_frames` измеряются в encoder-кадрах по 80 ms.
Последний вызов содержит `is_last=true`, `num_lookahead_frames=0`.
Ответ содержит вероятности спикеров и абсолютный `frame_offset` в mel-кадрах.
Look-ahead повторяется в начале следующего чанка и не считается обработанным.
`DELETE /v1/diarization/sessions/{session_id}` закрывает сессию досрочно.
По умолчанию максимум 32 сессии, TTL бездействия 900 секунд, до 4096 mel-кадров
на вызов. Клиент отправляет чанки одной сессии последовательно. Нумерация
спикеров сохраняется внутри сессии, между сессиями соответствия нет.

Профили оригинального checkpoint:

| Профиль | Chunk | Look-ahead | Ожидание аудио, без вычислений |
|---|---:|---:|---:|
| low_latency | 9 | 4 | 1.04 s |
| very_low_latency | 6 | 2 | 0.64 s |
| ultra_low_latency | 3 | 1 | 0.32 s |

Транспорт raw PCM/WebSocket в эту реализацию не входит. Для файлов используйте
`/v1/audio/diarizations`, где frontend уже включён.

## Использование из Python и CLI

```python
from nemotron_diarization_max import NemotronDiarization
from nemotron_diarization_max.audio import FeatureExtractor

model = NemotronDiarization.from_pretrained("models/nemotron")
frontend = FeatureExtractor()
inputs = frontend(audio_float32_mono_16khz, sampling_rate=16000)
result = model(inputs.input_features, inputs.attention_mask)
probabilities = result.probabilities  # [batch, mel_frames, 8]

# Поток: cache передаётся явно и принадлежит одной последовательности.
state = model.new_cache()
result = model(chunk_logmel, speaker_cache=state, num_lookahead_frames=4)
state = result.speaker_cache
```

```bash
nemotron-max diarize meeting.wav --model models/nemotron
```

## Проверки

Эталон — Transformers commit
`420dca92a4cf7d96dfbc1cc46f00375b0681ff8f`. В стабильном Transformers 5.12.1
класс этой модели отсутствует; для проверок закреплён исходный код выше.

```bash
# Сначала установите CPU-вариант torch, затем extras:
uv pip install torch --index-url https://download.pytorch.org/whl/cpu
uv pip install -e '.[test,reference]'
pytest -q

# Полный размер модели, воспроизводимые случайные веса:
python scripts/validate_reference.py --synthetic \
  --config models/nemotron/config.json --stream-steps 90 --output validation-local.json

# Сравнение на настоящем checkpoint, когда веса доступны:
python scripts/validate_reference.py --model models/nemotron \
  --output validation-trained.json
```

Тесты сравнивают логиты с upstream, частичные stacking-группы, padding/masks,
look-back/look-ahead, повторные обновления и сжатие cache, frontend, отсутствие
состояния между записями, загрузку shards и HTTP API.

Проверка на случайных весах подтверждает вычислительный перенос, но не качество
диаризации. Проверка DER на размеченной речи и сравнение настоящего checkpoint
пока не выполнены из-за недоступности файла весов. Производительность GPU и
работа в реальном времени не заявляются. См. [архитектуру](docs/architecture.md)
и отчёт полной CPU-проверки в `docs/validation-cpu-synthetic.json`.
