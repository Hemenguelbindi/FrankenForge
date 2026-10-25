# English is the fallback locale: every message used by the app must exist here.

## Toolbar
toolbar-open-file = Open file…
toolbar-open-folder = Open folder…
toolbar-opening = Opening { $path }…
toolbar-theme = Theme
toolbar-theme-dark = Dark
toolbar-theme-light = Light
toolbar-language = Interface language
dialog-open-file = Open a safetensors checkpoint file
dialog-open-folder = Open a checkpoint folder

## Empty state
welcome-title = Inspect a model checkpoint
welcome-body = Open a .safetensors file or a checkpoint folder. Only metadata is read: weights are not loaded and nothing is executed.

## Errors (the original error text is always shown under "Technical details")
error-open-title = Could not open { $path }
error-not-found = The file or folder does not exist (it may have been moved or deleted).
error-permission = Access denied. Check that you are allowed to read this file or folder.
error-io = The file could not be read from disk.
error-format = This is not a valid safetensors checkpoint, or the file is damaged. A folder must contain .safetensors files.
error-unknown = The checkpoint could not be opened.
error-crashed = Opening stopped unexpectedly because of an internal error.
error-details = Technical details
error-dismiss = Dismiss
startup-failed = FrankenForge could not open its window. This usually means the graphics driver or GPU is not available (for example over remote desktop).

## Model Overview
overview-title = Model Overview
overview-parameters = Parameters
overview-tensors = Tensors
overview-weight-types = Weight types
overview-components = Components
overview-config = Configuration (config.json)
count-thousand = { $value } K
count-million = { $value } M
count-billion = { $value } B
number-decimal-separator = { "." }

## Components
component-language = Language
component-audio = Audio
component-vision = Vision
component-other = Other
component-language-purpose = The main language part: repeated layers that read text tokens and predict the next one.
component-audio-purpose = Turns sound into features the language part can use. Found by the word "audio" in names.
component-vision-purpose = Turns images into features the language part can use. Found by "vision"/"visual" in names.
component-other-purpose = Weights whose role the names do not reveal: repeated blocks without language, audio or vision evidence, and tensors outside blocks.
card-blocks = { $blocks ->
        [one] { $blocks } block
       *[other] { $blocks } blocks
    } · { $params }
card-tensors = { $tensors ->
        [one] { $tensors } tensor
       *[other] { $tensors } tensors
    } · { $params }
card-loose = + { $count ->
        [one] { $count } tensor
       *[other] { $count } tensors
    } outside blocks
stack-blocks = { $name } — { $count ->
        [one] { $count } block
       *[other] { $count } blocks
    }

## Component tree
tree-explainer = Block: a repeated part of the model. Tensor: a named array of weights.
tree-block = Block { $index } · { $count ->
        [one] { $count } tensor
       *[other] { $count } tensors
    }
tree-loose = Outside blocks — { $count ->
        [one] { $count } tensor
       *[other] { $count } tensors
    }
tree-select = Select a component on the left.
grid-name = Name
grid-type = Type
grid-shape = Shape

## Search
search-hint = Search all tensors by name
search-matches = { $count ->
        [one] { $count } match
       *[other] { $count } matches
    }

## config.json
config-missing = No config.json in the checkpoint's folder.
config-source = Read from { $path }
config-too-large = config.json is too large to show (over 1 MB), so it was not read. The model itself opened normally.
config-not-json = config.json is not valid JSON, so it is not shown. The model itself opened normally.
config-unreadable = config.json could not be read from disk. The model itself opened normally.

## Tensor Inspector
inspector-title = Tensor Inspector
mode-simple = Simple
mode-expert = Expert
inspector-close = Close
inspector-belongs = Belongs to
inspector-block = { $stack }, block { $index }
inspector-outside-blocks = outside the repeated blocks
inspector-shape = Shape
inspector-dim = Dimension { $index }: { $size }
inspector-scalar = A single number (no dimensions).
inspector-shape-unknown = What these dimensions mean is not known for this tensor.
inspector-technical = Technical details
inspector-rule-basis = Rule basis
inspector-inferred-human = The purpose is guessed from the name; there are no exact architecture rules for it.
inspector-dtype = Data type
inspector-elements = Elements
inspector-size = Size
inspector-bias = { $title } — bias
size-bytes = { $value } bytes
size-kib = { $value } KiB
size-mib = { $value } MiB
size-gib = { $value } GiB

## Basis of an explanation
basis-confirmed = Confirmed
basis-confirmed-hint = Confirmed: config.json declares the "{ $arch }" architecture, and this exact tensor path is how that architecture names this part.
basis-inferred = Inferred
basis-inferred-hint = Inferred: a guess from the tensor's name. The architecture is not one with exact rules here, or config.json is missing, so the name could mean something else.
basis-unknown = Unknown
basis-unknown-hint = Unknown: no rule recognizes this name, so no purpose is claimed.

## Groups inside a block
group-attention = Attention
group-mlp = MLP
group-experts = Experts
group-normalization = Normalization
group-other = Other

## Dimensions
dim-hidden = model width (hidden_size)
dim-vocab = vocabulary size (vocab_size)
dim-heads-x-head-dim = attention heads × head size
dim-kv-heads-x-head-dim = key/value heads × head size
dim-head-dim = size of one attention head (head_dim)
dim-intermediate = MLP width (intermediate_size)
dim-moe-intermediate = expert MLP width (moe_intermediate_size)
dim-shared-intermediate = shared expert width (shared_expert_intermediate_size)
dim-experts = number of experts
dim-basis-confirmed-hint = Confirmed: config.json declares a size that matches this dimension.
dim-basis-inferred-hint = Inferred: the meaning follows from the tensor's name; config.json could not check it.
dim-basis-unknown-hint = Unknown: config.json contradicts this dimension, or nothing in the name or shape says what it means.

## Roles
role-token-embedding = Token embedding
role-token-embedding-desc = A table with one row per vocabulary token: it turns each input token into the vector the model works with.
role-position-embedding = Position embedding
role-position-embedding-desc = Adds information about where an element sits in the sequence or image.
role-attn-q = Attention: query (Q)
role-attn-q-desc = Turns each token into a "query" that is compared with the other tokens' keys to decide what to pay attention to.
role-attn-k = Attention: key (K)
role-attn-k-desc = Turns tokens into "keys" that the queries are compared against.
role-attn-v = Attention: value (V)
role-attn-v-desc = Turns tokens into "values": the information passed on once attention has decided where to look.
role-attn-o = Attention: output (O)
role-attn-o-desc = Combines the results of all attention heads back into the model width.
role-attn-qkv = Attention: fused Q, K and V
role-attn-qkv-desc = The query, key and value projections stored together in one matrix.
role-q-norm = Query normalization
role-q-norm-desc = Rescales each attention head's queries before they are compared with keys; keeps attention scores stable.
role-k-norm = Key normalization
role-k-norm-desc = Rescales each attention head's keys before they are compared with queries; keeps attention scores stable.
role-mlp-gate = MLP: gate projection
role-mlp-gate-desc = Part of the feed-forward network: decides, feature by feature, how much of the up projection passes through.
role-mlp-up = MLP: up projection
role-mlp-up-desc = Part of the feed-forward network: expands each token's vector into a wider space.
role-mlp-down = MLP: down projection
role-mlp-down-desc = Part of the feed-forward network: brings the result back to the model width.
role-mlp-gate-up = MLP: fused gate and up projections
role-mlp-gate-up-desc = The gate and up projections of the feed-forward network stored together in one matrix.
role-moe-router = MoE router
role-moe-router-desc = Scores every expert for each token; only the few best-scoring experts process that token.
role-expert-gate = Expert { $index }: gate projection
role-expert-up = Expert { $index }: up projection
role-expert-down = Expert { $index }: down projection
role-expert-gate-up = Expert { $index }: fused gate and up projections
role-expert-desc = One of several parallel feed-forward networks ("experts"); the router sends only some tokens to it. Which topics or skills an expert handles cannot be read from its weights.
role-shared-expert-gate = Shared expert: gate projection
role-shared-expert-up = Shared expert: up projection
role-shared-expert-down = Shared expert: down projection
role-shared-expert-gate-up = Shared expert: fused gate and up projections
role-shared-expert-desc = A feed-forward network that processes every token, in addition to the experts chosen by the router.
role-shared-expert-router = Shared expert gate
role-shared-expert-router-desc = Decides, for each token, how much of the shared expert's output is added.
role-norm = Normalization
role-norm-desc = Rescales vectors to a stable size; these numbers are per-feature scales.
role-final-norm = Final normalization
role-final-norm-desc = Normalizes the last hidden state before the output head.
role-output-head = Output head
role-output-head-desc = Turns the final vector into a score for every vocabulary token, i.e. predicts the next token.
role-projector = Projector
role-projector-desc = Maps features from another part of the model (an image or audio encoder) into the width the language part expects.
role-clip-bound = Clipping bound
role-clip-bound-desc = A single number that limits the range of values entering or leaving a layer, typically kept for quantized inference.
role-unknown = Purpose unknown
role-unknown-desc = No rule recognizes this name. Its full name, shape and type are shown below.

## Numeric profile
analyze-title = Numeric profile
analyze-hint = Reads only this tensor from disk, in small pieces, and computes statistics. Nothing is read until you press the button.
analyze-button = Analyze tensor
analyze-running = Analyzing… { $percent }%
analyze-empty = The tensor has no elements.
analyze-unsupported = Statistics are computed for floating-point tensors (F16, BF16, F32). This tensor is { $dtype }.
conclusion-nonfinite = Non-finite values: { $nan } NaN, { $pos_inf } positive infinity, { $neg_inf } negative infinity. This usually means a damaged file or a conversion error.
conclusion-all-zero = All { $count } elements are exactly zero.
conclusion-profile = Values range from { $min } to { $max }, with mean { $mean } and standard deviation { $std }. These numbers describe only the scale and spread of the stored values.
analyze-failed = The tensor could not be analyzed.
stat-count = Elements
stat-count-hint = How many numbers the tensor stores.
stat-min = Minimum
stat-min-hint = The smallest finite value.
stat-max = Maximum
stat-max-hint = The largest finite value.
stat-mean = Mean
stat-mean-hint = The average of the finite values.
stat-std = Standard deviation
stat-std-hint = How far values typically are from the mean: the tensor's scale.
stat-zeros = Zeros
stat-zeros-hint = Values that are exactly zero. Many zeros can mean pruning or unused rows.
stat-nan = NaN
stat-nan-hint = "Not a number" values. A healthy checkpoint has none.
stat-inf = Infinity (+/−)
stat-inf-hint = Infinite values. A healthy checkpoint has none.
stat-nonfinite-warning = This tensor contains NaN or infinite values. That usually means a damaged file or a bad conversion.
stat-no-finite = There are no finite values: every element is NaN or infinite.
stat-histogram = Distribution of finite values
stat-bin = { $from } … { $to }: { $count }
stat-explainer = These numbers describe how the stored values are spread: their typical scale (mean, standard deviation), their extremes and how many are exactly zero. They help to spot damaged files or unusual scaling. They do not show what the model knows, how well it works, or what any expert specializes in.
error-cancelled = The request was replaced by a newer one.

## Presentations
presentation-learn = Learn
presentation-structure = Model structure
presentation-technical = Technical data

## Learning route
learn-title = How a language model works
learn-intro = This short tour explains the idea behind the opened checkpoint. It is self-contained: no ML background is needed.
learn-scope = The tour uses only the file's structure. It does not run the model and does not show its real internal signals.
learn-facts-parts-label = Parts found
learn-facts-parts = { $parts }
learn-facts-no-parts = no part is labelled language, audio or vision
learn-facts-blocks-label = Repeated blocks
learn-facts-blocks = { $count }
learn-facts-no-blocks = none were recognized in this file
learn-facts-limits = The file shows structure and sizes. It does not show what the model has learned, why it gave a particular answer, or what any part "knows".
learn-route-title = From text to the next piece
learn-step-text-title = Text goes in
learn-step-text-body = You type a sentence, a question or a document.
learn-step-pieces-title = Text is split into pieces
learn-step-pieces-body = The model cuts the text into tokens: small pieces, often whole words or parts of words.
learn-step-numbers-title = Each piece becomes numbers
learn-step-numbers-body = Every token becomes a list of numbers, usually by looking it up in a learned table. The model works only with these numbers, never with the letters directly.
learn-step-blocks-title = Numbers pass through repeated blocks
learn-step-blocks-body = Each block mixes information between positions and transforms it. The model applies the same kind of block again and again.
learn-step-next-title = The model scores the next piece
learn-step-next-body = At the end, every possible next token gets a score. The model picks one, adds it to the text, and can repeat the process.
learn-explain-title = The ideas behind the route
explain-what-title = What the model does
explain-what-body = A language model is trained to continue text. Given the tokens so far, it estimates which token is likely next. It does not look up a finished answer: it computes scores from learned numbers and the current context.
explain-blocks-title = Why the same block repeats
explain-blocks-body = One block combines information from different positions and transforms it. Repeating the same kind of block lets each pass refine the representation. The file shows how many blocks exist; it does not show what each block has learned.
explain-attention-title = Attention: sharing context
explain-attention-body = Words often need information from other words. In the made-up example "Anna ... she", the word "she" may need the earlier name "Anna" to know who is meant. Attention computes how much each position should take from the others, then mixes that information in. It does not look up a fixed correspondence: the mixing weights are computed for the current context.
learn-attention-disclaimer = This is an illustration of the principle, not connections measured in the opened checkpoint.
explain-mlp-title = Additional processing: MLP and experts
explain-mlp-body = After attention, each position goes through a small network that transforms its numbers. A plain model uses one shared network. A mixture-of-experts model has several parallel networks and a router that sends each token to a few of them. The file shows that these units exist; it does not say what each one specializes in.
explain-norm-title = Normalization: keeping the scale steady
explain-norm-body = Numbers can grow or shrink over many steps. Normalization rescales them to a workable range before the next step, so the computation stays stable. The scale values come from the checkpoint itself; they are not a statement about the model's quality.
explain-tensor-title = What a tensor is
explain-tensor-body = A tensor is a named grid of learned numbers, usually a matrix. It is not a store of readable facts: a single value means nothing on its own, and the model's knowledge cannot be read from it like text. The file gives each tensor a name, shape and number type; the numbers are opened only in Technical data, when you ask for a numeric profile.
learn-diagram-block = Block { $index }
learn-diagram-more = …
learn-diagram-blocks-note = A repeated stack applies the same kind of block again and again.
learn-diagram-attn-name = Anna
learn-diagram-attn-pronoun = she
learn-diagram-attn-arrow = may refer back
learn-diagram-tensor = A tensor: a grid of learned numbers, not text.

## Model structure
structure-title = Model structure
structure-blocks = Repeated blocks
structure-parts = Parts of the file
structure-no-raw = This view explains what each part does. Raw names, shapes and numbers live in Technical data.
structure-explainer = Choose a block to open it. Each group below explains its purpose before any technical terms.
structure-stack = { $count ->
        [one] The same block repeats { $count } time
       *[other] The same block repeats { $count } times
    }
structure-block = Block { $index }
structure-block-parts = { $count ->
        [one] { $count } part
       *[other] { $count } parts
    }
structure-loose = Outside the repeated blocks — { $count ->
        [one] { $count } part
       *[other] { $count } parts
    }
structure-empty = No parts were recognized in this component.
structure-selected = Selected:
structure-open-technical = Open technical data
structure-bias = adjustment
structure-expert-index = unit { $index }
structure-select = Select a part on the left to see its structure.
structure-group-attention = Attention: sharing context
structure-group-attention-desc = Each piece compares itself with other pieces and mixes in the information it needs.
structure-group-mlp = Additional processing
structure-group-mlp-desc = After attention, a small network transforms each piece on its own.
structure-group-experts = Parallel processing units
structure-group-experts-desc = A router sends each piece to a few of several parallel units. The file does not say what each unit specializes in.
structure-group-normalization = Keeping the scale steady
structure-group-normalization-desc = These parts rescale numbers so the next step receives them in a workable range.
structure-group-other = Other parts
structure-group-other-desc = Parts that do not fit the groups above, or whose names the rules do not recognize.
structure-role-token-embedding = Turning text into numbers
structure-role-token-embedding-desc = A lookup table with one row per text piece. It turns each piece into the list of numbers the model works with.
structure-role-position-embedding = Where each piece sits
structure-role-position-embedding-desc = Adds order information so the model can tell positions apart.
structure-role-attn-q = What to look for
structure-role-attn-q-desc = Compares the current piece with earlier pieces to judge what is relevant.
structure-role-attn-k = What each piece offers
structure-role-attn-k-desc = Describes each piece so the comparison can judge relevance.
structure-role-attn-v = Information passed on
structure-role-attn-v-desc = Carries the information that is mixed into the current piece.
structure-role-attn-o = Combining what was gathered
structure-role-attn-o-desc = Combines the gathered information and returns it to the model's usual width.
structure-role-attn-qkv = Looking and carrying combined
structure-role-attn-qkv-desc = Stores the settings for looking and for carrying information in one part.
structure-role-q-norm = Keeping comparisons steady
structure-role-q-norm-desc = Rescales the comparison numbers so they stay in a workable range.
structure-role-k-norm = Keeping offers steady
structure-role-k-norm-desc = Rescales what each piece offers so comparisons stay in a workable range.
structure-role-mlp-gate = Filter for extra processing
structure-role-mlp-gate-desc = Decides, feature by feature, how much of the expanded result to keep.
structure-role-mlp-up = Widening the working space
structure-role-mlp-up-desc = Expands each piece into a wider space where the model can transform it.
structure-role-mlp-down = Back to the model width
structure-role-mlp-down-desc = Compresses the result back to the model's usual width.
structure-role-mlp-gate-up = Filter and widen together
structure-role-mlp-gate-up-desc = Filters and expands in one combined part.
structure-role-router = Choosing a unit
structure-role-router-desc = Chooses which parallel units process each piece.
structure-role-expert-gate = Parallel unit: filter
structure-role-expert-up = Parallel unit: widen
structure-role-expert-down = Parallel unit: compress
structure-role-expert-gate-up = Parallel unit: filter and widen
structure-role-expert-desc = One of several parallel units. The file does not say what it specializes in.
structure-role-shared-gate = Shared unit: filter
structure-role-shared-up = Shared unit: widen
structure-role-shared-down = Shared unit: compress
structure-role-shared-gate-up = Shared unit: filter and widen
structure-role-shared-desc = A unit that processes every piece, alongside the units chosen by the router.
structure-role-shared-router = Shared unit: amount
structure-role-shared-router-desc = Decides how much of the shared unit's result to add.
structure-role-norm = Keeping the scale steady
structure-role-norm-desc = Rescales the numbers so the next step works in a stable range.
structure-role-final-norm = Final scale check
structure-role-final-norm-desc = The last rescaling before the output.
structure-role-output = Scoring the next piece
structure-role-output-desc = Turns the final numbers into a score for every possible next text piece.
structure-role-projector = Connecting image or audio
structure-role-projector-desc = Translates features from an image or audio part into the language part's format.
structure-role-clip = Clipping limit
structure-role-clip-desc = A limit used when numbers are clipped to a fixed range.
structure-role-unknown = Unrecognized part
structure-role-unknown-desc = The name does not match known rules, so the purpose is unknown.