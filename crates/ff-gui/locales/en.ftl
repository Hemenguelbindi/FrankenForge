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

## Workspaces
workspace-inspector = Inspector
workspace-forge = Forge

## Errors from building
error-incompatible = The two models do not fit together: some tensors differ in name, type or shape.
error-unsupported-config = The model's config.json describes its layers in a way FrankenForge cannot update safely, so it refuses instead of producing a broken model.
error-recipe = The chosen parts could not be combined with these models.
error-verify = The written result did not match the plan, so it was rejected.

## Forge
forge-title = Forge: build a new model from two checkpoints
forge-intro = Choose a Base model and a Donor model, pick which blocks to take from the Donor, and FrankenForge creates a new model folder. The original models are only read and never changed.
forge-base = Base model
forge-base-hint = The foundation of the new model: everything you do not replace comes from it, including config.json and the tokenizer.
forge-donor = Donor model
forge-donor-hint = The source of parts: each block you select is copied from it into the same position in the Base.
forge-choose-folder = Choose folder…
forge-not-chosen = Not chosen
forge-dialog-base = Choose the Base model folder
forge-dialog-donor = Choose the Donor model folder
forge-compat-title = Structural compatibility
forge-structural-note = Structurally compatible means the tensors have the same names, types and shapes, so a block fits in place. It does not guarantee that the result loads in an inference program, keeps its quality, or produces sensible text.
forge-compat-empty = Choose both models to check which blocks fit.
forge-analyzing = Checking structural compatibility (only file headers are read)…
forge-analysis-failed = The models could not be compared
forge-tokenizer-identical = Tokenizers: identical. Both models split text into the same tokens.
forge-tokenizer-same-vocab = Tokenizers: the files differ, although the vocabulary is the same. Only identical tokenizers are allowed for now.
forge-tokenizer-different-vocab = Tokenizers: different vocabularies. The models split text into different tokens, so their blocks cannot be mixed.
forge-tokenizer-missing = Tokenizers: at least one model has no tokenizer.json, so this cannot be checked.
forge-stack-summary = { $stack }: { $compatible } of { $total } blocks are structurally compatible
forge-block = { $stack } · block { $index }
forge-row-base = Base
forge-row-donor = Donor
forge-schema-hint = Each model has its own row. Click a Donor block that can be transferred directly to take it into the same position of the Base; a seam joins the two.
forge-transfer-direct = can be transferred directly
forge-transfer-mapping = needs an explicit position mapping (not available yet)
forge-transfer-adapter = needs a trainable adapter (future experimental mode, not available)
forge-transfer-unsupported = transfer not supported: the structure differs
forge-legend-taken = selected: taken from the Donor
forge-cell-gap = no block at this position
forge-cell-replaced = replaced by Donor block { $index }
forge-cell-taken = goes to Base position { $index }
forge-select-none = Clear selection
forge-preview-title = Composition of the new model
forge-preview-empty = Select at least one structurally compatible block to see what the new model will consist of.
forge-preview-running = Planning the new model (metadata only)…
forge-preview-failed = A plan for this selection could not be made
forge-preview-blocks = Blocks taken from the Donor:
forge-preview-from-base = Tensors from the Base:
forge-preview-from-donor = Tensors from the Donor:
forge-preview-notes = Notes from the planner:
forge-output-title = Where to save the result
forge-output-hint = A new folder is created for the result. It must not exist yet and must not be inside the Base or Donor folder.
forge-output-location = Choose location…
forge-dialog-output = Choose where to create the new model folder
forge-output-name = New folder name:
forge-output-path = The result will be saved to:
forge-build = Build
forge-build-blocked = Before building:
forge-ready = Everything is ready to build.
forge-building = Build in progress: { $path }
forge-building-includes = It includes planning, copying, verification and writing the lock file. The core does not report which of these is running, so no progress is shown. The window stays responsive.
forge-build-title = Build
forge-stages-done = Completed:
forge-stage-plan = planning
forge-stage-copy = copying
forge-stage-verify = verification
forge-stage-lock = lock file
forge-locked = The models, blocks and output are locked until the build finishes. A running build cannot be stopped safely, so there is no Cancel button.
forge-build-ok = Built and verified
forge-build-ok-note = Verified means the written files were read back and every tensor matches the plan. It says nothing about whether the model loads in an inference program or answers well.
forge-result-model = Model folder:
forge-result-recipe = Recipe:
forge-result-lock = Lock file (hashes of sources and result):
forge-result-log = Build log:
forge-result-log-missing = could not be written (see the warning below)
forge-open-result = Open in Inspector
forge-build-refused = The build did not start. Nothing was written.
forge-build-failed = Build stopped
forge-build-failed-kept = Kept for diagnosis:
forge-need-base = Choose the Base model.
forge-need-donor = Choose the Donor model.
forge-same-model = Base and Donor are the same folder. Choose two different models.
forge-need-analysis = Wait until the compatibility check finishes.
forge-analysis-error = The compatibility check failed; see the message above.
forge-tokenizer-blocked = The tokenizers are not identical.
forge-need-blocks = Select at least one structurally compatible block.
forge-need-preview = Wait until the preview finishes.
forge-preview-error = The preview failed; see the message above.
forge-need-output = Choose where to save the result.
forge-bad-name = The folder name must be one plain name, without slashes, spaces at the ends, or "..".
forge-output-exists = This path already exists: { $path }. Choose another name; FrankenForge never overwrites anything.
forge-output-inside-source = The result would be saved inside the Base or Donor folder. Choose a location outside both models.
forge-output-unresolved = The chosen location cannot be checked (the drive may be unavailable or access is denied).
forge-step-base = Base
forge-step-donor = Donor
forge-step-blocks = Blocks
forge-step-build = Build
forge-step-done = done
forge-step-current = now
forge-step-waiting = waiting
forge-step-attention = needs attention
forge-mismatch-title = What does not match
forge-mismatch-no-blocks = No repeated block structure was recognized in at least one model, so its architecture is unknown to FrankenForge.
forge-mismatch-stack-only-base = { $stack }: only the Base has these blocks.
forge-mismatch-stack-only-donor = { $stack }: only the Donor has these blocks.
forge-mismatch-block-count = { $stack }: the Base has { $base } blocks, the Donor has { $donor }. Blocks without a partner at the same position would need an explicit mapping, which is not available yet.
forge-mismatch-block-shapes = { $stack }: { $count } blocks differ in tensor shape or type, for example block { $index }: { $example }
forge-mismatch-outer-shapes = { $count } tensors outside the blocks differ in shape, for example { $name }. This usually means a different hidden size or vocabulary.
forge-mismatch-outer-missing = Tensors outside the blocks: { $base } only in the Base, { $donor } only in the Donor.
forge-output-no-space = Not enough free disk space: the result needs about { $need }, the volume has { $available }. Free some space or choose another drive.
forge-disk-line = Disk: about { $need } needed, { $available } free.
forge-disk-unknown = Disk: about { $need } needed; free space could not be read ({ $why }).

## Hardware Preflight
workspace-preflight = Preflight
pf-title = Hardware Preflight
pf-intro = An estimate of the memory and disk an experiment needs, compared with what this computer reports. It is an estimate, not proof that a model loads or how fast it runs. Nothing is loaded or started here.
pf-hw-title = This computer
pf-hw-reading = Reading the hardware…
pf-hw-cpu = Processor:
pf-hw-ram = Memory (RAM):
pf-hw-gpu = Graphics card:
pf-hw-disks = Disks:
pf-hw-taken = Read at { $time } UTC. The values change while other programs run; read again before a long job.
pf-hw-reprobe = Read again
pf-unknown-why = unknown ({ $why })
pf-free-of = { $free } free of { $total }
pf-model-title = Model
pf-model-hint = Choose a model folder. Only the file headers and config.json are read; the sizes come from the actual tensors, not from the model's name.
pf-model-dialog = Choose a model folder
pf-model-failed = The model could not be read
pf-model-shape-unknown = The layer shape could not be read, so the cache and activations stay unknown: { $why }
pf-components = Components to load (untick parts you would not load, for example an image or audio encoder):
pf-component = { $name } — { $params } parameters, { $size } on disk
pf-settings-title = What to estimate
pf-scenario-inference = Running (inference)
pf-scenario-training = Training (full fine-tuning)
pf-placement = Where:
pf-placement-gpu = Graphics card
pf-placement-cpu = Processor and RAM (offload)
pf-precision = Precision:
pf-precision-int8 = 8-bit (only if the runtime supports it)
pf-context = Context length (tokens):
pf-batch = Batch size:
pf-checkpoints = Checkpoints kept:
pf-checkpoint-dir = Folder for checkpoints…
pf-training-note = The training estimate assumes full fine-tuning in f32 with the AdamW optimizer, as chosen for the first experiment. Nothing is trained here.
pf-result-title = Estimate
pf-result-need = Choose a model; the estimate appears when the hardware has been read.
pf-status-fits = Fits
pf-status-fits-desc = The estimate fits with a safety margin. This is not a guarantee: the runtime may need more than estimated.
pf-status-tight = Tight
pf-status-tight-desc = It may fit, but without a safety margin or only at the low end of the estimate. Consider the suggestions below.
pf-status-insufficient = Does not fit
pf-status-insufficient-desc = Even the lowest estimate is more than what is available. Change the settings or free resources before starting.
pf-status-unknown = Unknown
pf-status-unknown-desc = Something could not be measured or estimated, so a fit cannot be confirmed. Unknown never means it fits.
pf-res-vram = Graphics memory (VRAM)
pf-res-ram = Memory (RAM)
pf-res-disk = Disk
pf-check-line = needs { $need }, available { $available } —
pf-need-plus-unknown = { $need } plus unknown: { $parts }
pf-advice-title = What can be reduced
pf-parts-title = What the estimate consists of
pf-assumptions-title = Assumptions behind the numbers
pf-disclaimer = Preflight is an estimate from the file headers and the reported hardware. It does not prove that a model loads, runs correctly or runs fast.
pf-part-partial = result written to a .partial folder first
pf-part-reserve = reserve kept free
pf-part-weights = weights
pf-part-frozen = frozen weights
pf-part-trainable = trainable weights
pf-part-gradients = gradients
pf-part-optimizer = optimizer states (AdamW)
pf-part-kv = KV cache
pf-part-activations = activations
pf-part-logits = output logits
pf-part-adapters = adapters (none selected)
pf-part-runtime = runtime overhead
pf-part-checkpoints = checkpoints on disk
pf-assume-rename = The result is written to a .partial folder and renamed on the same disk, so only one copy exists at a time.
pf-assume-reserve = A reserve of free space is kept on the disk.
pf-assume-int8 = 8-bit counts only if the chosen runtime really supports it; this is not verified.
pf-assume-kv = The KV cache follows config.json (sliding window, shared layers); a runtime may allocate differently.
pf-assume-activations = Activations are a rough range, not a measurement.
pf-assume-runtime = Runtime overhead (driver context, workspaces) is a typical range, not measured for a specific runtime.
pf-assume-offload = Running on the processor avoids graphics memory but is much slower; it is not a free fix.
pf-assume-no-adapters = No adapters are loaded.
pf-assume-full-ft = Full fine-tuning in f32 with AdamW: every trained weight also needs a gradient and two optimizer states.
pf-assume-gpu-training = Training on the graphics card with this backend is not verified yet; only the processor path was tested.
pf-assume-single-gpu = Only the first reported graphics card is used.
pf-advice-batch = Reduce the batch size.
pf-advice-context = Reduce the context length.
pf-advice-components = Load fewer components (for example, leave out an image or audio encoder).
pf-advice-offload = Run on the processor and RAM instead: avoids graphics memory, but much slower.
pf-advice-disk = Free disk space or choose another drive.
pf-advice-reprobe = Read the hardware again; some values could not be measured.
forge-report-title = Build report
forge-report-behavior = Model behavior was NOT evaluated: nothing was run. The report only says where each part came from and how much the weights changed.
forge-report-file-verified = File: every tensor was read back after writing and its hash matches the plan.
forge-report-file-planned = Planned only: nothing is written yet, weights are not compared until the build.
forge-report-stack = { $stack }: { $before } blocks before, { $after } after
forge-report-status-unchanged = unchanged
forge-report-status-shifted = moved to a new position
forge-report-status-replaced = replaced ({ $total }/{ $total } tensors)
forge-report-status-partial = partially replaced ({ $replaced }/{ $total } tensors)
forge-report-status-added = added (no such block in the Base)
forge-report-status-generated = generated (experimental)
forge-report-delta = weight change { $percent }
forge-report-delta-unknown = weight change unknown: { $reason }
forge-report-delta-not-computed = weight change not computed
forge-report-removed-blocks = Removed blocks: { $names }
forge-report-removed-tensors = Removed tensors: { $names }
forge-report-outside-unchanged = { $name }: from { $source }, unchanged
forge-report-outside-replaced = { $name }: replaced from { $source } { $delta }
forge-report-outside-added = { $name }: added from { $source }
forge-report-outside-generated = { $name }: generated (experimental)
forge-report-config-none = config.json: the Base has none.
forge-report-config-unchanged = config.json: unchanged.
forge-report-config-changed = config.json changes:
forge-report-warnings = Warnings:
forge-report-experimental = EXPERIMENTAL: contains generated tensors that nothing can run. This is not a working model.
forge-diff-title = Where the models differ
forge-diff-intro = How far the weights of each Donor block are from the same block of the Base. This shows where training made the Donor different, so you can see which blocks are worth taking instead of guessing.
forge-diff-need-analysis = Choose both models first; the difference is measured for blocks that are built the same way in both.
forge-diff-start = Measure the difference
forge-diff-start-hint = Reads the weights of the matching blocks of both models (the models are not changed). For a small model this takes seconds.
forge-diff-running = Reading weights… { $percent }%
forge-diff-stop = Stop
forge-diff-stopped = Stopped. Nothing was measured.
forge-diff-failed = The difference could not be measured
forge-diff-none = No block is built the same way in both models: the matrices have other sizes or other parts. There is nothing to compare weight by weight, and such a block cannot be put into the Base directly: its numbers would not fit the Base's internal format.
forge-diff-summary = From { $min } (block { $min_block }) to { $max } (block { $max_block })
forge-diff-click = Click a block to take it from the Donor.
forge-diff-block = block { $index }
forge-diff-unknown = unknown
forge-diff-most = most changed: { $part } { $value }
forge-diff-not-compared = { $count } blocks are built differently and were not compared.
forge-diff-how-to-read = How to read it: 0% means the block is the same as in the Base. The larger the number, the more the Donor's training changed this block. If all blocks differ by about the same amount, the changes are spread over the whole model, and one transplanted block carries only a small part of them. The number says how much the weights moved, not whether the model gets better: that is only visible by running the model.
graph-flow = { $from } → { $to }
graph-norms = { $count } normalizations keep the numbers in range

net-words = Word table
net-output-words = Output: words
net-per-layer = Hints for every block
net-bridge = Bridge
net-block-tip = Block { $index } · { $params } parameters
net-legend-size = Circle size = number of parameters
net-legend-shared = Orange ring: the block reuses the attention memory of another block
net-legend-controls = Wheel: zoom · drag: move · click: what is inside
inside-summary = What the program found in this model
inside-largest = Largest part: { $name } ({ $share }% of the model)
inside-entry = Data enters here: { $name }
inside-exit = The next word comes out here: { $name }
inside-most-linked = Most links: { $name }
inside-pick = Click a circle on the graph to see what is inside it.
inside-size = { $params } parameters · { $share }% of the model
inside-words = One row per word (token) of the vocabulary of the model, { $rows } rows. Each row is the list of numbers the model uses for that word; words with a similar meaning get similar rows. Type a word to see its neighbours.
inside-output-words = Turns the result of the last block back into a score for each of the { $rows } words; the word with the highest score is the next word. Type a word to see which words this table treats as similar.
inside-per-layer = A lookup table, not a computing part: for every word it holds a separate small set of numbers for each of the { $blocks } blocks, handed to that block directly. It can be huge because it has a row for every word of the vocabulary. Type a word and choose a block to see which words get a similar hint there.
inside-block = Block { $index } of { $stack }: it reads the stream of numbers, works on it and adds its result back.
inside-block-kind = Kind from config.json: { $kind }
inside-shared-memory = This block keeps no attention memory of its own; it reuses the memory of an earlier block.
inside-neurons-hint = The feed-forward part has thousands of neurons. Each one, when it fires, pushes the next word in a direction. The program can read that direction as words from the weights alone.
inside-neurons-button = Which words do its neurons push toward?
inside-neurons-none = This block has no recognised feed-forward part, so its neurons cannot be read as words.
inside-bridge = Translates the numbers of this part into the numbers the language part works with, so that what was seen or heard can be talked about.
inside-word-label = Word:
inside-for-block = block
inside-words-button = Show similar words
inside-reading = Reading the weights…
inside-stop = Stop
inside-stopped = Stopped.
inside-failed = Could not read this part
inside-near = Next to “{ $word }”:
inside-near-note = The number is how similar the rows are (1 = the same direction). Read from the weights only; nothing was run.
inside-not-one-token = This model splits the word into several pieces, so it has no single row. Pieces it starts with:
inside-neurons-title = The 16 strongest neurons of this block and the words each pushes toward
inside-neuron = neuron { $index }:
inside-neurons-note = Read from the weights only (projection onto the vocabulary), nothing was run. Many neurons read as a mix of unrelated pieces: that is normal, a neuron rarely means one word. This shows where a neuron pushes, not everything it does.
net-per-layer-part = Part of the hints path
