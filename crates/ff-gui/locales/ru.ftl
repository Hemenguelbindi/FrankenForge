# Russian. Missing messages fall back to English.

## Toolbar
toolbar-open-file = Открыть файл…
toolbar-open-folder = Открыть папку…
toolbar-opening = Открываю { $path }…
toolbar-theme = Тема
toolbar-theme-dark = Тёмная
toolbar-theme-light = Светлая
toolbar-language = Язык интерфейса
dialog-open-file = Открыть файл checkpoint в формате safetensors
dialog-open-folder = Открыть папку checkpoint

## Empty state
welcome-title = Просмотр checkpoint модели
welcome-body = Откройте файл .safetensors или папку checkpoint. Читаются только метаданные: веса не загружаются, ничего не запускается.

## Errors
error-open-title = Не удалось открыть { $path }
error-not-found = Файл или папка не существует (возможно, они перемещены или удалены).
error-permission = Доступ запрещён. Проверьте, что у вас есть права на чтение этого файла или папки.
error-io = Не удалось прочитать файл с диска.
error-format = Это не корректный checkpoint safetensors, или файл повреждён. В папке должны быть файлы .safetensors.
error-unknown = Не удалось открыть checkpoint.
error-crashed = Открытие неожиданно прервалось из-за внутренней ошибки.
error-details = Технические подробности
error-dismiss = Закрыть
startup-failed = FrankenForge не смог открыть окно. Обычно это значит, что недоступен графический драйвер или GPU (например, при работе через удалённый рабочий стол).

## Model Overview
overview-title = Обзор модели
overview-parameters = Параметры
overview-tensors = Тензоры
overview-weight-types = Типы весов
overview-components = Компоненты
overview-config = Конфигурация (config.json)
count-thousand = { $value } тыс.
count-million = { $value } млн
count-billion = { $value } млрд
number-decimal-separator = { "," }

## Components
component-language = Язык
component-audio = Аудио
component-vision = Изображения
component-other = Прочее
component-language-purpose = Основная языковая часть: повторяющиеся слои, которые читают токены текста и предсказывают следующий.
component-audio-purpose = Превращает звук в признаки, понятные языковой части. Найдено по слову «audio» в именах.
component-vision-purpose = Превращает изображения в признаки, понятные языковой части. Найдено по словам «vision»/«visual» в именах.
component-other-purpose = Веса, роль которых не видна из имён: повторяющиеся блоки без признаков языка, аудио или изображений и тензоры вне блоков.
card-blocks = { $blocks ->
        [one] { $blocks } блок
        [few] { $blocks } блока
       *[many] { $blocks } блоков
    } · { $params }
card-tensors = { $tensors ->
        [one] { $tensors } тензор
        [few] { $tensors } тензора
       *[many] { $tensors } тензоров
    } · { $params }
card-loose = + { $count ->
        [one] { $count } тензор
        [few] { $count } тензора
       *[many] { $count } тензоров
    } вне блоков
stack-blocks = { $name } — { $count ->
        [one] { $count } блок
        [few] { $count } блока
       *[many] { $count } блоков
    }

## Component tree
tree-explainer = Block — повторяющаяся часть модели; tensor — именованный массив весов.
tree-block = Блок { $index } · { $count ->
        [one] { $count } тензор
        [few] { $count } тензора
       *[many] { $count } тензоров
    }
tree-loose = Вне блоков — { $count ->
        [one] { $count } тензор
        [few] { $count } тензора
       *[many] { $count } тензоров
    }
tree-select = Выберите компонент слева.
grid-name = Имя
grid-type = Тип
grid-shape = Форма

## Search
search-hint = Поиск по именам всех тензоров
search-matches = { $count ->
        [one] { $count } совпадение
        [few] { $count } совпадения
       *[many] { $count } совпадений
    }

## config.json
config-missing = В папке checkpoint нет config.json.
config-source = Прочитано из { $path }
config-too-large = config.json слишком большой для показа (больше 1 МБ) и не был прочитан. Сама модель открылась нормально.
config-not-json = config.json не является корректным JSON и не показан. Сама модель открылась нормально.
config-unreadable = Не удалось прочитать config.json с диска. Сама модель открылась нормально.

## Tensor Inspector
inspector-title = Инспектор тензора
mode-simple = Простой
mode-expert = Эксперт
inspector-close = Закрыть
inspector-belongs = Принадлежит
inspector-block = { $stack }, блок { $index }
inspector-outside-blocks = вне повторяющихся блоков
inspector-shape = Форма
inspector-dim = Измерение { $index }: { $size }
inspector-scalar = Одно число (без измерений).
inspector-shape-unknown = Смысл этих измерений для этого тензора неизвестен.
inspector-technical = Технические сведения
inspector-rule-basis = Основание правила
inspector-inferred-human = Назначение предполагается по имени; точных правил архитектуры нет.
inspector-dtype = Тип данных
inspector-elements = Элементов
inspector-size = Размер
inspector-bias = { $title } — смещение (bias)
size-bytes = { $value } байт
size-kib = { $value } КиБ
size-mib = { $value } МиБ
size-gib = { $value } ГиБ

## Basis of an explanation
basis-confirmed = Подтверждено
basis-confirmed-hint = Подтверждено: config.json объявляет архитектуру «{ $arch }», и именно этим путём эта архитектура называет данную часть.
basis-inferred = Предположение
basis-inferred-hint = Предположение по имени тензора. Для этой архитектуры здесь нет точных правил или нет config.json, поэтому имя может означать и другое.
basis-unknown = Неизвестно
basis-unknown-hint = Неизвестно: ни одно правило не распознаёт это имя, поэтому назначение не утверждается.

## Groups inside a block
group-attention = Внимание (attention)
group-mlp = MLP
group-experts = Эксперты
group-normalization = Нормализация
group-other = Прочее

## Dimensions
dim-hidden = ширина модели (hidden_size)
dim-vocab = размер словаря (vocab_size)
dim-heads-x-head-dim = головы внимания × размер головы
dim-kv-heads-x-head-dim = головы key/value × размер головы
dim-head-dim = размер одной головы внимания (head_dim)
dim-intermediate = ширина MLP (intermediate_size)
dim-moe-intermediate = ширина MLP эксперта (moe_intermediate_size)
dim-shared-intermediate = ширина общего эксперта (shared_expert_intermediate_size)
dim-experts = число экспертов
dim-basis-confirmed-hint = Подтверждено: config.json объявляет размер, совпадающий с этим измерением.
dim-basis-inferred-hint = Предположение: смысл следует из имени тензора; config.json не позволил его проверить.
dim-basis-unknown-hint = Неизвестно: config.json противоречит этому измерению, или ни имя, ни форма не говорят, что оно означает.

## Roles
role-token-embedding = Эмбеддинг токенов
role-token-embedding-desc = Таблица, в которой на каждый токен словаря приходится одна строка: она превращает входной токен в вектор, с которым работает модель.
role-position-embedding = Позиционный эмбеддинг
role-position-embedding-desc = Добавляет сведения о том, где элемент находится в последовательности или изображении.
role-attn-q = Внимание: запрос (Q)
role-attn-q-desc = Превращает каждый токен в «запрос», который сравнивается с ключами других токенов, чтобы решить, на что обратить внимание.
role-attn-k = Внимание: ключ (K)
role-attn-k-desc = Превращает токены в «ключи», с которыми сравниваются запросы.
role-attn-v = Внимание: значение (V)
role-attn-v-desc = Превращает токены в «значения» — сведения, которые передаются дальше, когда внимание решило, куда смотреть.
role-attn-o = Внимание: выход (O)
role-attn-o-desc = Собирает результаты всех голов внимания обратно в ширину модели.
role-attn-qkv = Внимание: объединённые Q, K и V
role-attn-qkv-desc = Проекции запроса, ключа и значения, хранящиеся вместе в одной матрице.
role-q-norm = Нормализация запросов
role-q-norm-desc = Масштабирует запросы каждой головы внимания перед сравнением с ключами, чтобы оценки внимания оставались устойчивыми.
role-k-norm = Нормализация ключей
role-k-norm-desc = Масштабирует ключи каждой головы внимания перед сравнением с запросами, чтобы оценки внимания оставались устойчивыми.
role-mlp-gate = MLP: проекция-затвор (gate)
role-mlp-gate-desc = Часть сети прямого распространения: решает для каждого признака, какая доля up-проекции проходит дальше.
role-mlp-up = MLP: расширяющая проекция (up)
role-mlp-up-desc = Часть сети прямого распространения: расширяет вектор каждого токена в более широкое пространство.
role-mlp-down = MLP: сужающая проекция (down)
role-mlp-down-desc = Часть сети прямого распространения: возвращает результат к ширине модели.
role-mlp-gate-up = MLP: объединённые gate и up
role-mlp-gate-up-desc = Проекции gate и up сети прямого распространения, хранящиеся вместе в одной матрице.
role-moe-router = Маршрутизатор MoE
role-moe-router-desc = Оценивает каждого эксперта для каждого токена; токен обрабатывают только несколько экспертов с лучшей оценкой.
role-expert-gate = Эксперт { $index }: проекция gate
role-expert-up = Эксперт { $index }: проекция up
role-expert-down = Эксперт { $index }: проекция down
role-expert-gate-up = Эксперт { $index }: объединённые gate и up
role-expert-desc = Одна из нескольких параллельных сетей прямого распространения («экспертов»); маршрутизатор отправляет к ней только часть токенов. По весам нельзя узнать, какие темы или навыки обрабатывает эксперт.
role-shared-expert-gate = Общий эксперт: проекция gate
role-shared-expert-up = Общий эксперт: проекция up
role-shared-expert-down = Общий эксперт: проекция down
role-shared-expert-gate-up = Общий эксперт: объединённые gate и up
role-shared-expert-desc = Сеть прямого распространения, которая обрабатывает каждый токен в дополнение к экспертам, выбранным маршрутизатором.
role-shared-expert-router = Затвор общего эксперта
role-shared-expert-router-desc = Решает для каждого токена, какая доля выхода общего эксперта добавляется.
role-norm = Нормализация
role-norm-desc = Приводит векторы к устойчивому масштабу; эти числа — масштабы по каждому признаку.
role-final-norm = Финальная нормализация
role-final-norm-desc = Нормализует последнее скрытое состояние перед выходной головой.
role-output-head = Выходная голова
role-output-head-desc = Превращает итоговый вектор в оценку для каждого токена словаря, то есть предсказывает следующий токен.
role-projector = Проектор
role-projector-desc = Переводит признаки из другой части модели (кодировщика изображений или звука) в ширину, которую ожидает языковая часть.
role-clip-bound = Граница отсечения
role-clip-bound-desc = Одно число, ограничивающее диапазон значений на входе или выходе слоя; обычно хранится для квантованного вывода.
role-unknown = Назначение неизвестно
role-unknown-desc = Ни одно правило не распознаёт это имя. Полное имя, форма и тип показаны ниже.

## Numeric profile
analyze-title = Численный профиль
analyze-hint = Читает с диска только этот тензор, небольшими частями, и считает статистику. Пока вы не нажмёте кнопку, ничего не читается.
analyze-button = Анализировать тензор
analyze-running = Анализ… { $percent }%
analyze-empty = В тензоре нет элементов.
analyze-unsupported = Статистика считается для тензоров с плавающей точкой (F16, BF16, F32). Этот тензор — { $dtype }.
conclusion-nonfinite = Неконечные значения: NaN — { $nan }, +Inf — { $pos_inf }, −Inf — { $neg_inf }. Обычно это означает повреждённый файл или ошибку конвертации.
conclusion-all-zero = Все элементы ({ $count }) в точности равны нулю.
conclusion-profile = Значения лежат в диапазоне от { $min } до { $max }; среднее — { $mean }, стандартное отклонение — { $std }. Эти числа описывают только масштаб и разброс хранимых значений.
analyze-failed = Не удалось проанализировать тензор.
stat-count = Элементов
stat-count-hint = Сколько чисел хранит тензор.
stat-min = Минимум
stat-min-hint = Наименьшее конечное значение.
stat-max = Максимум
stat-max-hint = Наибольшее конечное значение.
stat-mean = Среднее
stat-mean-hint = Среднее арифметическое конечных значений.
stat-std = Стандартное отклонение
stat-std-hint = Насколько значения обычно отличаются от среднего: масштаб тензора.
stat-zeros = Нули
stat-zeros-hint = Значения, в точности равные нулю. Много нулей может означать прореживание или неиспользуемые строки.
stat-nan = NaN
stat-nan-hint = Значения «не число». В исправном checkpoint их нет.
stat-inf = Бесконечность (+/−)
stat-inf-hint = Бесконечные значения. В исправном checkpoint их нет.
stat-nonfinite-warning = В тензоре есть NaN или бесконечные значения. Обычно это значит, что файл повреждён или неудачно сконвертирован.
stat-no-finite = Конечных значений нет: каждый элемент — NaN или бесконечность.
stat-histogram = Распределение конечных значений
stat-bin = { $from } … { $to }: { $count }
stat-explainer = Эти числа описывают, как распределены хранимые значения: их типичный масштаб (среднее, стандартное отклонение), крайние значения и сколько из них ровно ноль. Они помогают заметить повреждённые файлы или необычное масштабирование. Они не показывают, что знает модель, насколько хорошо она работает или на чём специализируется какой-либо эксперт.
error-cancelled = Запрос заменён более новым.

## Presentations
presentation-learn = Обучение
presentation-structure = Устройство модели
presentation-technical = Технические данные

## Learning route
learn-title = Как работает языковая модель
learn-intro = Этот короткий маршрут объясняет идею открытой модели. Подготовка в ML не нужна.
learn-scope = Маршрут использует только структуру файла. Модель не запускается, реальные внутренние сигналы не показываются.
learn-facts-parts-label = Найденные части
learn-facts-parts = { $parts }
learn-facts-no-parts = ни одна часть не подписана как языковая, аудио или зрение
learn-facts-blocks-label = Повторяющиеся блоки
learn-facts-blocks = { $count }
learn-facts-no-blocks = в этом файле их распознать не удалось
learn-facts-limits = Файл показывает структуру и размеры. Он не показывает, чему модель научилась, почему дала конкретный ответ и что «знает» каждая часть.
learn-route-title = От текста к следующей части
learn-step-text-title = Текст на входе
learn-step-text-body = Вы вводите предложение, вопрос или документ.
learn-step-pieces-title = Текст делится на части
learn-step-pieces-body = Модель разбивает текст на токены — небольшие куски, часто целые слова или их части.
learn-step-numbers-title = Каждая часть превращается в числа
learn-step-numbers-body = Каждый токен становится списком чисел — обычно через поиск в обученной таблице. Дальше модель работает только с числами, не с буквами напрямую.
learn-step-blocks-title = Числа проходят через повторяющиеся блоки
learn-step-blocks-body = Каждый блок смешивает информацию между позициями и преобразует её. Модель снова и снова применяет блок одного и того же типа.
learn-step-next-title = Модель оценивает следующую часть
learn-step-next-body = В конце каждый возможный следующий токен получает оценку. Модель выбирает один, добавляет его к тексту и может повторить процесс.
learn-explain-title = Идеи, стоящие за маршрутом
explain-what-title = Что делает модель
explain-what-body = Языковая модель учится продолжать текст. По уже имеющимся токенам она оценивает, какой токен вероятнее следующим. Она не ищет готовый ответ, а вычисляет оценки по обученным числам и текущему контексту.
explain-blocks-title = Зачем нужен повтор блоков
explain-blocks-body = Один блок объединяет информацию из разных позиций и преобразует её. Повторение одного и того же типа блока позволяет каждому проходу уточнять представление. Файл показывает, сколько блоков есть; он не показывает, чему научился каждый блок.
explain-attention-title = Attention: обмен контекстом
explain-attention-body = Словам часто нужна информация из других слов. В придуманном примере «Анна ... она» слову «она» может понадобиться ранее упомянутое имя «Анна», чтобы понять, о ком речь. Attention вычисляет, сколько каждая позиция должна взять у остальных, и смешивает эту информацию. Он не ищет фиксированного соответствия: веса смешивания вычисляются для текущего контекста.
learn-attention-disclaimer = Это иллюстрация принципа, а не связи, измеренные в открытой модели.
explain-mlp-title = Дополнительная обработка: MLP и эксперты
explain-mlp-body = После attention каждая позиция проходит через небольшую сеть, которая преобразует её числа. В обычной модели сеть одна и общая. В модели с экспертами есть несколько параллельных сетей и маршрутизатор, который направляет каждый токен лишь в некоторые из них. Файл показывает, что такие блоки есть; он не говорит, на чём специализируется каждый из них.
explain-norm-title = Normalization: контроль масштаба
explain-norm-body = По мере прохождения многих шагов числа могут становиться очень большими или очень маленькими. Normalization приводит их к рабочему диапазону перед следующим шагом, чтобы вычисления оставались устойчивыми. Значения масштаба берутся из самого файла модели и не являются оценкой её качества.
explain-tensor-title = Что такое тензор
explain-tensor-body = Тензор — это именованная сетка обученных чисел, обычно матрица. Это не хранилище читаемых фактов: отдельное значение ничего не значит само по себе, и знания модели нельзя прочитать из него как текст. Файл даёт тензору имя, форму и тип чисел; сами числа открываются только в «Технических данных», когда вы запрашиваете числовой профиль.
learn-diagram-block = Блок { $index }
learn-diagram-more = …
learn-diagram-blocks-note = Повторяющийся стек снова и снова применяет блок одного и того же типа.
learn-diagram-attn-name = Анна
learn-diagram-attn-pronoun = она
learn-diagram-attn-arrow = может относиться назад
learn-diagram-tensor = Тензор: сетка обученных чисел, а не текст.

## Model structure
structure-title = Устройство модели
structure-blocks = Повторяющиеся блоки
structure-parts = Части файла
structure-no-raw = Здесь объясняется, что делает каждая часть. Сырые имена, формы и числа — в «Технических данных».
structure-explainer = Выберите блок, чтобы раскрыть его. Каждая группа сначала объясняет назначение простыми словами, без технических терминов.
structure-stack = { $count ->
        [one] Блок повторяется { $count } раз
        [few] Блок повторяется { $count } раза
       *[many] Блок повторяется { $count } раз
    }
structure-block = Блок { $index }
structure-block-parts = { $count ->
        [one] { $count } часть
        [few] { $count } части
       *[many] { $count } частей
    }
structure-loose = Вне повторяющихся блоков — { $count ->
        [one] { $count } часть
        [few] { $count } части
       *[many] { $count } частей
    }
structure-empty = В этой части ничего не распознано.
structure-selected = Выбрано:
structure-open-technical = Открыть технические данные
structure-bias = поправка
structure-expert-index = модуль { $index }
structure-select = Выберите часть слева, чтобы увидеть её устройство.
structure-group-attention = Attention: обмен контекстом
structure-group-attention-desc = Каждая часть сравнивает себя с другими и примешивает нужную ей информацию.
structure-group-mlp = Дополнительная обработка
structure-group-mlp-desc = После attention небольшая сеть отдельно преобразует каждую часть.
structure-group-experts = Параллельные модули
structure-group-experts-desc = Маршрутизатор направляет каждую часть в несколько из многих параллельных модулей. Файл не говорит, на чём специализируется каждый модуль.
structure-group-normalization = Контроль масштаба
structure-group-normalization-desc = Эти части перемасштабируют числа, чтобы следующий шаг получил их в рабочем диапазоне.
structure-group-other = Другие части
structure-group-other-desc = Части, не подходящие к группам выше, или те, чьи имена не распознаны правилами.
structure-role-token-embedding = Превращение текста в числа
structure-role-token-embedding-desc = Таблица поиска: по одной строке на каждый кусок текста. Она превращает кусок в список чисел, с которыми работает модель.
structure-role-position-embedding = Где находится каждая часть
structure-role-position-embedding-desc = Добавляет информацию о порядке, чтобы модель различала позиции.
structure-role-attn-q = Что искать
structure-role-attn-q-desc = Сравнивает текущий кусок с предыдущими, чтобы понять, что важно.
structure-role-attn-k = Что предлагает каждая часть
structure-role-attn-k-desc = Описывает кусок так, чтобы сравнение могло оценить его важность.
structure-role-attn-v = Передаваемая информация
structure-role-attn-v-desc = Несёт информацию, которая примешивается к текущему куску.
structure-role-attn-o = Объединение собранного
structure-role-attn-o-desc = Собирает полученную информацию и возвращает её к обычной ширине модели.
structure-role-attn-qkv = Поиск и передача вместе
structure-role-attn-qkv-desc = Хранит настройки поиска и передачи информации в одной части.
structure-role-q-norm = Стабильность сравнений
structure-role-q-norm-desc = Приводит числа сравнения к рабочему диапазону.
structure-role-k-norm = Стабильность предложений
structure-role-k-norm-desc = Приводит то, что предлагает каждая часть, к рабочему диапазону, чтобы сравнения оставались устойчивыми.
structure-role-mlp-gate = Фильтр дополнительной обработки
structure-role-mlp-gate-desc = По каждому признаку решает, какую часть расширенного результата сохранить.
structure-role-mlp-up = Расширение рабочего пространства
structure-role-mlp-up-desc = Разворачивает каждый кусок в более широкое пространство, где модель его преобразует.
structure-role-mlp-down = Возврат к ширине модели
structure-role-mlp-down-desc = Сжимает результат обратно к обычной ширине модели.
structure-role-mlp-gate-up = Фильтр и расширение вместе
structure-role-mlp-gate-up-desc = Фильтрует и расширяет в одной объединённой части.
structure-role-router = Выбор модуля
structure-role-router-desc = Решает, какие параллельные модули обрабатывают каждый кусок.
structure-role-expert-gate = Параллельный модуль: фильтр
structure-role-expert-up = Параллельный модуль: расширение
structure-role-expert-down = Параллельный модуль: сжатие
structure-role-expert-gate-up = Параллельный модуль: фильтр и расширение
structure-role-expert-desc = Один из нескольких параллельных модулей. Файл не говорит, на чём он специализируется.
structure-role-shared-gate = Общий модуль: фильтр
structure-role-shared-up = Общий модуль: расширение
structure-role-shared-down = Общий модуль: сжатие
structure-role-shared-gate-up = Общий модуль: фильтр и расширение
structure-role-shared-desc = Модуль, который обрабатывает каждый кусок вместе с модулями, выбранными маршрутизатором.
structure-role-shared-router = Общий модуль: доля
structure-role-shared-router-desc = Решает, какую долю результата общего модуля добавить.
structure-role-norm = Контроль масштаба
structure-role-norm-desc = Приводит числа к устойчивому диапазону, чтобы следующий шаг работал стабильно.
structure-role-final-norm = Последняя проверка масштаба
structure-role-final-norm-desc = Последнее перемасштабирование перед выходом.
structure-role-output = Оценка следующей части
structure-role-output-desc = Превращает итоговые числа в оценку для каждого возможного следующего куска текста.
structure-role-projector = Связь с изображением или звуком
structure-role-projector-desc = Переводит признаки из части изображения или звука в формат языковой части.
structure-role-clip = Предел отсечения
structure-role-clip-desc = Предел, который используется, когда числа отсекаются до фиксированного диапазона.
structure-role-unknown = Нераспознанная часть
structure-role-unknown-desc = Имя не совпадает с известными правилами, поэтому назначение неизвестно.

## Workspaces
workspace-inspector = Инспектор
workspace-forge = Кузница

## Errors from building
error-incompatible = Модели не подходят друг к другу: часть тензоров отличается по имени, типу или форме.
error-unsupported-config = config.json модели описывает слои так, что FrankenForge не может безопасно его обновить, поэтому сборка отклонена, чтобы не получить сломанную модель.
error-recipe = Выбранные части не удалось совместить с этими моделями.
error-verify = Записанный результат не совпал с планом, поэтому он отклонён.

## Forge
forge-title = Кузница: сборка новой модели из двух checkpoint
forge-intro = Выберите базовую модель и модель-донор, отметьте блоки, которые нужно взять у донора, и FrankenForge создаст папку новой модели. Исходные модели только читаются и никогда не изменяются.
forge-base = Базовая модель (Base)
forge-base-hint = Основа новой модели: всё, что вы не заменяете, берётся из неё, включая config.json и токенизатор.
forge-donor = Модель-донор (Donor)
forge-donor-hint = Источник частей: каждый выбранный блок копируется из неё на то же место в базовой модели.
forge-choose-folder = Выбрать папку…
forge-not-chosen = Не выбрано
forge-dialog-base = Выберите папку базовой модели
forge-dialog-donor = Выберите папку модели-донора
forge-compat-title = Структурная совместимость
forge-structural-note = Структурно совместимый блок — это блок с теми же именами, типами и формами тензоров, поэтому он встаёт на место. Это не гарантирует, что результат загрузится в программе для запуска модели, сохранит качество или будет выдавать осмысленный текст.
forge-compat-empty = Выберите обе модели, чтобы проверить, какие блоки подходят.
forge-analyzing = Проверяю структурную совместимость (читаются только заголовки файлов)…
forge-analysis-failed = Не удалось сравнить модели
forge-tokenizer-identical = Токенизаторы: идентичны. Обе модели одинаково делят текст на токены.
forge-tokenizer-same-vocab = Токенизаторы: файлы различаются, хотя словарь одинаковый. Пока разрешены только идентичные токенизаторы.
forge-tokenizer-different-vocab = Токенизаторы: разные словари. Модели по-разному делят текст на токены, поэтому их блоки нельзя смешивать.
forge-tokenizer-missing = Токенизаторы: хотя бы у одной модели нет tokenizer.json, поэтому проверить это нельзя.
forge-stack-summary = { $stack }: структурно совместимых блоков — { $compatible } из { $total }
forge-block = { $stack } · блок { $index }
forge-row-base = База
forge-row-donor = Донор
forge-schema-hint = У каждой модели своя строка. Нажмите на блок донора, который можно перенести напрямую, — он займёт ту же позицию в базовой модели, и их соединит шов.
forge-transfer-direct = можно перенести напрямую
forge-transfer-mapping = нужно явное соответствие позиций (пока недоступно)
forge-transfer-adapter = нужен обучаемый адаптер (будущий экспериментальный режим, недоступно)
forge-transfer-unsupported = перенос не поддерживается: структура отличается
forge-legend-taken = выбран: берётся у донора
forge-cell-gap = на этой позиции нет блока
forge-cell-replaced = заменяется блоком донора { $index }
forge-cell-taken = встаёт на позицию { $index } базовой модели
forge-select-none = Снять выбор
forge-preview-title = Состав новой модели
forge-preview-empty = Отметьте хотя бы один структурно совместимый блок, чтобы увидеть, из чего будет состоять новая модель.
forge-preview-running = Планирую новую модель (только метаданные)…
forge-preview-failed = Не удалось составить план для этого выбора
forge-preview-blocks = Блоки, взятые у донора:
forge-preview-from-base = Тензоров из базовой модели:
forge-preview-from-donor = Тензоров из донора:
forge-preview-notes = Замечания планировщика:
forge-output-title = Куда сохранить результат
forge-output-hint = Для результата создаётся новая папка. Она не должна существовать и не должна находиться внутри папки базовой модели или донора.
forge-output-location = Выбрать место…
forge-dialog-output = Выберите, где создать папку новой модели
forge-output-name = Имя новой папки:
forge-output-path = Результат будет сохранён в:
forge-build = Собрать
forge-build-blocked = Перед сборкой:
forge-ready = Всё готово к сборке.
forge-building = Сборка выполняется: { $path }
forge-building-includes = Она включает планирование, копирование, проверку и создание lock-файла. Ядро не сообщает, какой из этих этапов идёт сейчас, поэтому прогресс не показывается. Окно при этом не зависает.
forge-build-title = Сборка
forge-stages-done = Завершено:
forge-stage-plan = планирование
forge-stage-copy = копирование
forge-stage-verify = проверка
forge-stage-lock = lock-файл
forge-locked = Модели, блоки и место сохранения заблокированы до конца сборки. Безопасно остановить идущую сборку нельзя, поэтому кнопки «Отмена» нет.
forge-build-ok = Собрано и проверено
forge-build-ok-note = «Проверено» значит, что записанные файлы перечитаны и каждый тензор совпадает с планом. Это ничего не говорит о том, загрузится ли модель в программе для запуска и хорошо ли она отвечает.
forge-result-model = Папка модели:
forge-result-recipe = Рецепт:
forge-result-lock = Lock-файл (хэши исходников и результата):
forge-result-log = Журнал сборки:
forge-result-log-missing = не удалось записать (см. предупреждение ниже)
forge-open-result = Открыть в инспекторе
forge-build-refused = Сборка не запущена. Ничего не записано.
forge-build-failed = Сборка остановлена
forge-build-failed-kept = Сохранено для диагностики:
forge-need-base = Выберите базовую модель.
forge-need-donor = Выберите модель-донор.
forge-same-model = Базовая модель и донор — одна и та же папка. Выберите две разные модели.
forge-need-analysis = Дождитесь окончания проверки совместимости.
forge-analysis-error = Проверка совместимости не удалась, см. сообщение выше.
forge-tokenizer-blocked = Токенизаторы не идентичны.
forge-need-blocks = Отметьте хотя бы один структурно совместимый блок.
forge-need-preview = Дождитесь окончания предпросмотра.
forge-preview-error = Предпросмотр не удался, см. сообщение выше.
forge-need-output = Выберите, куда сохранить результат.
forge-bad-name = Имя папки должно быть одним простым именем: без косых черт, пробелов по краям и "..".
forge-output-exists = Этот путь уже существует: { $path }. Выберите другое имя: FrankenForge никогда ничего не перезаписывает.
forge-output-inside-source = Результат оказался бы внутри папки базовой модели или донора. Выберите место вне обеих моделей.
forge-output-unresolved = Не удаётся проверить выбранное место (диск может быть недоступен или нет прав доступа).
forge-step-base = База
forge-step-donor = Донор
forge-step-blocks = Блоки
forge-step-build = Сборка
forge-step-done = готово
forge-step-current = сейчас
forge-step-waiting = ожидает
forge-step-attention = требует внимания
forge-mismatch-title = Что не совпадает
forge-mismatch-no-blocks = Хотя бы в одной модели не распознана повторяющаяся структура блоков, поэтому её архитектура неизвестна FrankenForge.
forge-mismatch-stack-only-base = { $stack }: такие блоки есть только в базовой модели.
forge-mismatch-stack-only-donor = { $stack }: такие блоки есть только у донора.
forge-mismatch-block-count = { $stack }: в базовой модели блоков — { $base }, у донора — { $donor }. Для блоков без пары на той же позиции нужно явное соответствие, а оно пока недоступно.
forge-mismatch-block-shapes = { $stack }: блоков с другой формой или типом тензоров — { $count }, например блок { $index }: { $example }
forge-mismatch-outer-shapes = Тензоров вне блоков с другой формой — { $count }, например { $name }. Обычно это значит, что отличается скрытый размер или словарь.
forge-mismatch-outer-missing = Тензоры вне блоков: только в базовой модели — { $base }, только у донора — { $donor }.
forge-output-no-space = Недостаточно места на диске: результату нужно около { $need }, а на томе свободно { $available }. Освободите место или выберите другой диск.
forge-disk-line = Диск: нужно около { $need }, свободно { $available }.
forge-disk-unknown = Диск: нужно около { $need }; свободное место прочитать не удалось ({ $why }).

## Hardware Preflight
workspace-preflight = Предпроверка
pf-title = Предпроверка оборудования
pf-intro = Оценка памяти и места на диске, которые нужны эксперименту, в сравнении с тем, что сообщает этот компьютер. Это оценка, а не доказательство того, что модель загрузится или будет работать быстро. Здесь ничего не загружается и не запускается.
pf-hw-title = Этот компьютер
pf-hw-reading = Опрашиваю оборудование…
pf-hw-cpu = Процессор:
pf-hw-ram = Память (RAM):
pf-hw-gpu = Видеокарта:
pf-hw-disks = Диски:
pf-hw-taken = Прочитано в { $time } UTC. Значения меняются, пока работают другие программы; перед долгой задачей опросите заново.
pf-hw-reprobe = Опросить заново
pf-unknown-why = неизвестно ({ $why })
pf-free-of = свободно { $free } из { $total }
pf-model-title = Модель
pf-model-hint = Выберите папку модели. Читаются только заголовки файлов и config.json; размеры берутся из реальных тензоров, а не из названия модели.
pf-model-dialog = Выберите папку модели
pf-model-failed = Не удалось прочитать модель
pf-model-shape-unknown = Не удалось прочитать форму слоёв, поэтому кэш и активации остаются неизвестными: { $why }
pf-components = Загружаемые компоненты (снимите отметку с частей, которые не нужны, например энкодер изображений или звука):
pf-component = { $name } — { $params } параметров, { $size } на диске
pf-settings-title = Что оценивать
pf-scenario-inference = Запуск (inference)
pf-scenario-training = Обучение (полное fine-tuning)
pf-placement = Где:
pf-placement-gpu = Видеокарта
pf-placement-cpu = Процессор и RAM (offload)
pf-precision = Точность:
pf-precision-int8 = 8 бит (только если runtime это поддерживает)
pf-context = Длина контекста (токенов):
pf-batch = Размер батча:
pf-checkpoints = Хранить чекпоинтов:
pf-checkpoint-dir = Папка для чекпоинтов…
pf-training-note = Оценка обучения предполагает полное fine-tuning в f32 с оптимизатором AdamW, как выбрано для первого эксперимента. Здесь ничего не обучается.
pf-result-title = Оценка
pf-result-need = Выберите модель; оценка появится, когда оборудование будет опрошено.
pf-status-fits = Помещается
pf-status-fits-desc = Оценка помещается с запасом. Это не гарантия: runtime может потребовать больше, чем оценено.
pf-status-tight = Впритык
pf-status-tight-desc = Может поместиться, но без запаса или только при нижней границе оценки. Посмотрите советы ниже.
pf-status-insufficient = Не помещается
pf-status-insufficient-desc = Даже нижняя граница оценки больше доступного. Измените настройки или освободите ресурсы перед запуском.
pf-status-unknown = Неизвестно
pf-status-unknown-desc = Что-то не удалось измерить или оценить, поэтому подтвердить, что всё поместится, нельзя. «Неизвестно» никогда не значит «помещается».
pf-res-vram = Видеопамять (VRAM)
pf-res-ram = Память (RAM)
pf-res-disk = Диск
pf-check-line = нужно { $need }, доступно { $available } —
pf-need-plus-unknown = { $need } плюс неизвестно: { $parts }
pf-advice-title = Что можно уменьшить
pf-parts-title = Из чего состоит оценка
pf-assumptions-title = Допущения, на которых основаны числа
pf-disclaimer = Предпроверка — это оценка по заголовкам файлов и сведениям об оборудовании. Она не доказывает, что модель загрузится, будет работать правильно или быстро.
pf-part-partial = результат сначала записывается в папку .partial
pf-part-reserve = запас свободного места
pf-part-weights = веса
pf-part-frozen = замороженные веса
pf-part-trainable = обучаемые веса
pf-part-gradients = градиенты
pf-part-optimizer = состояния оптимизатора (AdamW)
pf-part-kv = KV-кэш
pf-part-activations = активации
pf-part-logits = выходные logits
pf-part-adapters = адаптеры (не выбраны)
pf-part-runtime = накладные расходы runtime
pf-part-checkpoints = чекпоинты на диске
pf-assume-rename = Результат пишется в папку .partial и переименовывается на том же диске, поэтому одновременно существует одна копия.
pf-assume-reserve = На диске оставляется запас свободного места.
pf-assume-int8 = 8 бит учитываются, только если выбранный runtime действительно это поддерживает; это не проверено.
pf-assume-kv = KV-кэш считается по config.json (скользящее окно, общие слои); runtime может выделять память иначе.
pf-assume-activations = Активации — грубый диапазон, а не измерение.
pf-assume-runtime = Накладные расходы runtime (контекст драйвера, рабочие буферы) — типичный диапазон, не измеренный для конкретного runtime.
pf-assume-offload = Работа на процессоре не требует видеопамяти, но значительно медленнее; это не бесплатное решение.
pf-assume-no-adapters = Адаптеры не загружаются.
pf-assume-full-ft = Полное fine-tuning в f32 с AdamW: каждому обучаемому весу нужны ещё градиент и два состояния оптимизатора.
pf-assume-gpu-training = Обучение на видеокарте с этим backend пока не проверено; проверен только путь на процессоре.
pf-assume-single-gpu = Используется только первая из найденных видеокарт.
pf-advice-batch = Уменьшите размер батча.
pf-advice-context = Уменьшите длину контекста.
pf-advice-components = Загружайте меньше компонентов (например, без энкодера изображений или звука).
pf-advice-offload = Запустите на процессоре и RAM: видеопамять не нужна, но работа намного медленнее.
pf-advice-disk = Освободите место на диске или выберите другой диск.
pf-advice-reprobe = Опросите оборудование заново: часть значений измерить не удалось.
forge-report-title = Отчёт сборки
forge-report-behavior = Поведение модели НЕ проверялось: ничего не запускалось. Отчёт говорит только, откуда взята каждая часть и насколько изменились веса.
forge-report-file-verified = Файл: после записи каждый тензор перечитан, его хеш совпадает с планом.
forge-report-file-planned = Только план: ещё ничего не записано, веса сравниваются при сборке.
forge-report-stack = { $stack }: было блоков { $before }, стало { $after }
forge-report-status-unchanged = без изменений
forge-report-status-shifted = сдвинут на новую позицию
forge-report-status-replaced = заменён ({ $total }/{ $total } тензоров)
forge-report-status-partial = заменён частично ({ $replaced }/{ $total } тензоров)
forge-report-status-added = добавлен (в Базе такого блока нет)
forge-report-status-generated = сгенерирован (экспериментально)
forge-report-delta = изменение весов { $percent }
forge-report-delta-unknown = изменение весов неизвестно: { $reason }
forge-report-delta-not-computed = изменение весов не вычислялось
forge-report-removed-blocks = Удалённые блоки: { $names }
forge-report-removed-tensors = Удалённые тензоры: { $names }
forge-report-outside-unchanged = { $name }: из { $source }, без изменений
forge-report-outside-replaced = { $name }: заменён из { $source } { $delta }
forge-report-outside-added = { $name }: добавлен из { $source }
forge-report-outside-generated = { $name }: сгенерирован (экспериментально)
forge-report-config-none = config.json: в Базе его нет.
forge-report-config-unchanged = config.json: без изменений.
forge-report-config-changed = Изменения config.json:
forge-report-warnings = Предупреждения:
forge-report-experimental = ЭКСПЕРИМЕНТАЛЬНО: есть сгенерированные тензоры, которые ничто не умеет запускать. Это не рабочая модель.
