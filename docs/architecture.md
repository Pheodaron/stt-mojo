# Пайплайн NeMo и его перенос на Go

Изучены исходники NeMo Speech, ревизия
[`8eb458b8`](https://github.com/NVIDIA-NeMo/Speech/tree/8eb458b8ee304143426be5070d3f55fb3b5fb736):

- [`sortformer_diar_models.py`](https://github.com/NVIDIA-NeMo/Speech/blob/8eb458b8ee304143426be5070d3f55fb3b5fb736/nemo/collections/asr/models/sortformer_diar_models.py): `process_signal`, `forward_streaming`, `forward_streaming_step`, `_diarize_output_processing`.
- [`sortformer_modules.py`](https://github.com/NVIDIA-NeMo/Speech/blob/8eb458b8ee304143426be5070d3f55fb3b5fb736/nemo/collections/asr/modules/sortformer_modules.py): `streaming_feat_loader`, `streaming_update`, `_compress_spkcache`.
- [`vad_utils.py`](https://github.com/NVIDIA-NeMo/Speech/blob/8eb458b8ee304143426be5070d3f55fb3b5fb736/nemo/collections/asr/parts/utils/vad_utils.py): `predlist_to_timestamps`.

`SortformerEncLabelModel` обслуживает несколько архитектур. Параметры именно
Nemotron-3-Diarization берутся из checkpoint. Соответствие весов и frontend
проверяются по [реализации Transformers](https://github.com/huggingface/transformers/tree/420dca92a4cf7d96dfbc1cc46f00375b0681ff8f/src/transformers/models/nemotron3_diarization).

## Как проходит одна запись

В NeMo обработка длинного файла использует `forward_streaming`, даже если
файл уже целиком записан. Синхронный `streaming_update` подходит этому режиму.
Единица состояния — одна запись; отдельные чанки продолжают её историю.

1. `process_signal` получает waveform и строит признаки. В режиме chunked
   inference нормализация всей записи по пику пропускается. Для нашего checkpoint:
   preemphasis 0.97, Hann 400, FFT 512, hop 160, 128 Slaney mel-фильтров и log,
   без нормализации mel по всей записи.
2. `streaming_feat_loader` выделяет новые кадры с левым и правым контекстом.
   Профиль файла: 340 новых encoder-кадров и 40 кадров справа — 27.2 и 3.2 s.
   На один encoder-кадр приходится восемь mel-кадров по 10 ms.
3. `_call_pre_encode` выполняет feature stacking и проекцию. Вход энкодера:
   `[speaker cache | FIFO | left context | new chunk | right context]`.
4. Энкодер и голова вычисляют весь этот вход. У Nemotron: 31 Transformer-блок,
   RoPE, проекция и convolution/subpixel upsampling до разрешения 10 ms,
   затем классификатор восьми спикеров. Позиции RoPE начинаются с нуля на каждом
   вызове, включая позиции cache/FIFO. Авторегрессионного KV cache здесь нет.
5. Вероятности головы усредняются по восьми кадрам для обновления состояния.
   Только новые embeddings добавляются в FIFO; контекст туда не попадает.
6. При переполнении FIFO часть кадров переносится в speaker cache. Cache
   сжимается до 264 кадров: log scores предпочитают уверенную речь одного спикера,
   strong/weak boost сохраняет представительство спикеров, latest boost помогает
   новым кадрам. Резервируются silence slots с обученным `silence_embeds`.
   Результат сгруппирован по спикерам с сохранением порядка кадров внутри группы.
7. Из предсказаний вырезаются cache, FIFO и контекст. Padding последнего чанка
   исключается из результата. Состояние последнего чанка можно не обновлять.
8. `_diarize_output_processing` отдельно строит интервалы: sigmoid probabilities,
   пороги onset/offset, padding границ, удаление короткой речи и объединение
   коротких пауз, затем RTTM. Это отдельный от вызова модели этап.

До первой компрессии NeMo обновляет оценки всех хронологических кадров cache
по текущему предсказанию. После компрессии сохранённые оценки остаются прежними:
кадры уже сгруппированы по спикерам. Этот порядок сохранён в Go.

## Что оставлено в сервисе

| Возможность | Решение |
|---|---|
| Декодирование, выбор каналов, ресэмплинг | На входе уже mono PCM 16 kHz float32 |
| Препроцессинг | Сохранён, с непрерывной preemphasis и перекрытием STFT |
| Чанки, контекст, stacking, padding | Сохранены |
| FIFO, сжатие speaker cache, learned silence | Сохранены |
| Энкодер, RoPE, голова, high-resolution output | MAX; выход — логиты каждые 10 ms |
| Несколько одновременно активных записей, async batching | Убраны; вызывающая сторона отправляет одну запись за раз |
| Реестр сессий, очереди, блокировки модели, фоновые задачи | Отсутствуют |
| RTTM, сглаживание интервалов, сопоставление словам ASR | Внешнее приложение; API возвращает логиты |
| Training, SpecAugment, permutations, DDP, CUDA graphs, activity head | Не входят в этот inference-путь |

`oom_safe_feature_extraction` в NeMo делит **batch**, а не один длинный файл
по времени; это видно из `start_idx/end_idx`. Go читает waveform и признаки
порциями, чтобы не хранить двухчасовую запись и всю mel-матрицу.

## Архитектура Go

```text
Run(ctx, io.Reader, emit)
  featureReader → chunkReader → MAX → speakerCache → emit(Result)
```

- `frontend.go`: последовательное чтение PCM, preemphasis и log-mel.
  `readFrames` возвращает число кадров, действительную длину и `io.EOF`.
- `chunks.go`: чтение нужного числа признаков, вызов `Model.Embed`, сохранение
  контекста и выдача следующего окна. Проекция оставшегося контекста повторно
  не вычисляется. В полях reader хранятся только данные чтения.
- `pipeline.go`: один цикл — получить окно, вызвать модель, обновить cache,
  вырезать ответ и отдать его. EOF завершает запись; отдельного `Finish` нет.
- `cache.go`: две последовательности embeddings и политика отбора кадров.
  Кандидат имеет явные поля `speaker`, `frame`, `score`; sentinel-индексы не нужны.
- `maxruntime`: синхронные `Embed` и `Predict`. Вся работа с C-указателями
  изолирована здесь. `Predict` принимает действительную длину вместо массива
  булевых масок: padding всегда находится в конце.
- `server.go`: адаптер gRPC → `io.Reader` и отправка `Result`. В нём нет алгоритма
  обработки аудио или состояния модели.

Один `Pipeline` создаётся при запуске. Каждый `Run` создаёт локальные frontend,
chunk reader и speaker cache. Последовательные записи независимы. Параллельный
вызов не входит в контракт: синхронизации и планировщика нет. Дополнительная
горутина отправки в примере gRPC-клиента нужна для двустороннего flow control одной
записи; инференс остаётся последовательным.

Буферы FFT и чтения переиспользуются. FFT выполняется Gonum; mel-фильтры обходят
только ненулевые коэффициенты. Hann, mel weights и inverse RoPE frequencies
экспортируются как float32; sin/cos готовятся один раз для максимального окна.
MAX получает Go-буферы только на время синхронного cgo-вызова. Графы и веса
загружаются один раз, Python во время обслуживания запросов не запускается.

При равных scores Go сохраняет исходный порядок; `torch.topk` порядок равенств
не гарантирует. Различия округления у порогов также могут менять дискретный выбор.
Численные тесты проверяют разные конечные оценки, silence slots и политику NeMo.
EOF, неполные группы, транспортные границы и независимость последовательных
записей проверяются отдельно.
