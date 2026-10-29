# Nemotron 3 Diarization · Go + MAX CPU

Сервис принимает декодированный **mono PCM 16 kHz, float32 little endian** и
возвращает логиты восьми спикеров с шагом 10 ms. Одна запись обрабатывается
последовательно; следующую запись клиент отправляет после завершения предыдущей.

```text
PCM → Go log-mel → чанки → MAX C API → Go FIFO/speaker cache → логиты
```

[Разбор NeMo, выбранные фичи и архитектура Go](docs/architecture.md).
Основной цикл — [Pipeline.Run](internal/diarization/pipeline.go). Он работает
с обычным `io.Reader`: одинаково для локального файла и gRPC-потока. Нет реестра
сессий, блокировок модели, планировщика запросов и фоновых workers.

## Запуск

Linux x86-64, Go ≥ 1.24, C compiler. Python 3.12 нужен для подготовки MAX SDK
и экспорта графов; сервер вызывает MAX напрямую через cgo без Python.
Checkpoint: `config.json`, `processor_config.json`, `model.safetensors` или shards.

```bash
uv venv
uv pip install --python .venv/bin/python -e .
.venv/bin/python -m nemotron_diarization_max.export \
  --model models/nemotron --output models/nemotron-cpu
make build
export MAX_ROOT="$(.venv/bin/python -c 'import site; print(site.getsitepackages()[0])')/modular"
bin/nemotron-server --model models/nemotron-cpu --listen 0.0.0.0:50051
```

MAX 26.7.0, CPU, float32. MEF-графы зависят от версии SDK и платформы CPU.
Серверу нужны библиотеки SDK и экспортированные файлы. При переносе SDK задайте
`MAX_ROOT` и `LD_LIBRARY_PATH="$MAX_ROOT/lib"`. `make` принимает `GO`, `PYTHON`,
`MAX_SITE`, `MAX_ROOT`. Если стандартный cache недоступен для записи, задайте
`XDG_CACHE_HOME` и `UV_CACHE_DIR` в доступный каталог.

## gRPC

[Контракт](api/diarization/v1/diarization.proto): один bidirectional RPC
`diarization.v1.Diarization/Diarize` — одна запись.

1. Отправляйте `AudioChunk.pcm_f32le`: PCM без WAV-заголовка, целое число float32
   сэмплов в сообщении. Рекомендуемый размер — 64 KiB, лимит сообщения — 1 MiB.
2. Одновременно читайте `Prediction`: `logits_f32le` — row-major
   `[num_frames, num_speakers]`; `start_frame` — абсолютный индекс кадра × 10 ms.
3. В конце вызовите `CloseSend()` и дочитайте ответы до EOF.

Пример клиента с необходимым двусторонним чтением/отправкой:

```bash
bin/diarize --input recording.f32 --output logits.f32
```

`logit > 0` соответствует вероятности `> 0.5`. Несколько спикеров могут говорить
одновременно. Возвращается `floor(N/160)` кадров для N сэмплов. Минимальная
длина — 160 сэмплов; NaN/Inf и неполные float32 отклоняются. Декодирование,
ресэмплинг, интервалы спикеров и ASR выполняет вызывающее приложение.

Профиль длинного файла: 27.2 s новых данных и 3.2 s правого контекста на шаг,
FIFO 40, speaker cache 264, update period 300. Двухчасовая запись проходит
примерно за 265 **шагов модели**, без накопления всей записи и всех результатов.
Это число шагов, а не оценка времени выполнения. Speaker ID согласованы внутри
одного RPC. Отмена прекращает чтение и следующие шаги; текущий MAX-вызов синхронный.

## Проверки

```bash
make test
uv pip install --python .venv/bin/python torch --index-url https://download.pytorch.org/whl/cpu
uv pip install --python .venv/bin/python -e '.[test,reference]'
curl -fL https://raw.githubusercontent.com/NVIDIA-NeMo/Speech/8eb458b8ee304143426be5070d3f55fb3b5fb736/nemo/collections/asr/modules/sortformer_modules.py \
  -o /tmp/sortformer_modules.py
.venv/bin/python scripts/prepare_go_fixtures.py \
  --output /tmp/nemotron-go-fixture --nemo-source /tmp/sortformer_modules.py
NEMOTRON_TEST_BUNDLE=/tmp/nemotron-go-fixture make test
.venv/bin/pytest -q
```

Тесты сравнивают frontend и MAX/gRPC с Transformers, а FIFO и сжатие cache —
с оригинальными методами NeMo. Двухчасовой тест читает PCM и выполняет настоящий
preprocessing, используя дешёвую тестовую модель. Он проверяет число кадров и
размеры окон. Fixtures используют случайные веса маленькой модели. Настоящие
веса NVIDIA, качество на речи и скорость полной модели пока не проверены.
[Результаты проверок](docs/validation-go-cpu.json).
