"""Reserved special-token inventory for Electra.

These tokens are allocated at the front of the shared tokenizer vocabulary
(spec-addendum A2). They implement the typed structured-state serialization
from docs.md section 10 and the question/candidate wrappers used by the
decision branches.
"""

META_TOKENS = [
    "<pad>",
    "<unk>",
    "<bos>",
    "<eos>",
    "<sep>",
    "<cls>",
    "<mask>",
]

STATE_STRUCT_TOKENS = [
    "<state>",
    "</state>",
    "<obj>",
    "<end_obj>",
    "<key>",
    "<array>",
    "<end_array>",
]

PRIMITIVE_TYPE_TOKENS = [
    "<num>",
    "<str>",
    "<bool_true>",
    "<bool_false>",
    "<null>",
]

QUESTION_TOKENS = [
    "<q>",
    "</q>",
    "<instruction>",
    "</instruction>",
    "<candidate>",
    "</candidate>",
]

TEXT_STATE_TOKENS = [
    "<doc>",
    "</doc>",
    "<field>",
    "<value>",
]

CORRUPTION_TOKENS = [
    "<corrupt>",
]

SPECIAL_TOKENS: list[str] = (
    META_TOKENS
    + STATE_STRUCT_TOKENS
    + PRIMITIVE_TYPE_TOKENS
    + QUESTION_TOKENS
    + TEXT_STATE_TOKENS
    + CORRUPTION_TOKENS
)

PAD = "<pad>"
UNK = "<unk>"
BOS = "<bos>"
EOS = "<eos>"
SEP = "<sep>"
CLS = "<cls>"
MASK = "<mask>"

SPECIAL_TOKEN_IDS: dict[str, int] = {tok: i for i, tok in enumerate(SPECIAL_TOKENS)}
NUM_SPECIAL_TOKENS = len(SPECIAL_TOKENS)
