# Hidden-Message Extraction

Evaluates whether language models can correctly extract hidden messages from text given explicit extraction rules.

## What This Pack Measures

This pack tests a model's ability to follow precise text-analysis instructions. Each case provides carrier text with a hidden message encoded using a specific scheme, along with explicit rules for extraction. The model must apply the rules exactly and output only the extracted message.

This measures:
- Instruction following precision
- Character-level text manipulation
- Pattern recognition and rule application
- Ability to identify when no message is present (false positive resistance)

## Schemes

### Acrostic (12 cases)
The first letter of each line spells the hidden message. Carrier texts are written as natural-sounding multi-line passages (poetry, prose, instructions). Messages range from 4-7 characters.

**Example:**
- Text lines start with: **H**appiness, **E**very, **L**ove, **P**eople
- Hidden message: `HELP`

### Index-of-Word (12 cases)
A specified character position from each word (or a subset of words) forms the message. Variants include:
- 1st letter of each word
- 2nd letter of each word
- Last letter of each word
- 1st or 3rd letter of every other word

**Example:**
- Rule: "Read the last letter of each word"
- Text: `epic zero said done`
- Hidden message: `CODE`

### Punctuation Mapping (10 cases)
Punctuation marks encode the message through various mappings:
- Symbol-to-letter mapping (comma=A, period=B, semicolon=C, etc.)
- Comma counting per sentence mapped to alphabet position (1=A, 2=B, ...)
- Exclamation mark counting per sentence
- Binary encoding (comma=0, semicolon=1, convert 8-bit groups to ASCII)

**Example:**
- Rule: "Each comma represents A, each period represents B, each semicolon represents C"
- Text: `Sure; fine; okay, great.`
- Punctuation sequence: `;` `;` `,` `.`
- Hidden message: `CCAB`

### Hidden Character (12 cases)
Characters hidden within text using character-level rules. Two subtypes:

**Subtype A: Capitalized letters in lowercase text (6 cases)**
The carrier text is entirely lowercase except for specific letters that spell the hidden message. The model must read only the capitalized letters from left to right.

**Example:**
- Text: `tHe sun rosE sLowly as People gathered below`
- Capitalized letters: **H**, **E**, **L**, **P**
- Hidden message: `HELP`

**Subtype B: Nth character of text (6 cases)**
Every Nth character of the string (counting from position 1, including all characters) spells the hidden message. Different cases use different values of N (3, 4, 5, 6).

**Example:**
- Rule: "Read every 3rd character"
- Text: `xxAxxCxxE`
- Positions 3, 6, 9: **A**, **C**, **E**
- Hidden message: `ACE`

### No-Hint (10 cases)
The model receives a carrier text with no extraction rule -- only the instruction "There may be a hidden message encoded in it." The model must independently discover the encoding scheme and extract the message, or output `NONE` if no message is present. Cases reuse carrier texts from other schemes (acrostic, index_of_word, hidden_character) plus control texts with no hidden message. This tests the model's ability to recognize steganographic patterns without guidance.

### No-Message Controls (10 cases)
Regular text with no hidden message under the given extraction rule. The model must output exactly `NONE`. These test false-positive resistance -- the model should not hallucinate a message where none exists.

### Noise Variants (8 cases)
Valid hidden messages with added noise that the model must filter out:
- Blank lines inserted between message-carrying lines
- Header/footer lines to ignore
- Parenthetical text to skip
- Comment lines (starting with `#`) to ignore
- ALL-CAPS words to skip
- Lines of dashes to ignore
- Punctuation inside square brackets to ignore

## Scoring

| Outcome | Score | Label |
|---------|-------|-------|
| Exact match (after normalization) | 1.0 | CORRECT |
| Control case correctly returns NONE | 1.0 | CORRECT |
| Partial match (substring containment) | 0.5 | PARTIAL |
| Control case returns a message | 0.0 | FALSE_POSITIVE |
| No match | 0.0 | INCORRECT |

Normalization strips whitespace and converts to uppercase before comparison.

## Case Distribution

| Scheme | Count |
|--------|-------|
| acrostic | 12 |
| index_of_word | 12 |
| punctuation_mapping | 10 |
| hidden_character | 12 |
| no_hint | 10 |
| no_message_control | 10 |
| noise_variant | 8 |
| **Total** | **74** |

## Files

- `pack.yaml` - Pack configuration and system prompt
- `cases.yaml` - All 74 test cases with carrier text and expected messages
- `grader.py` - Scoring logic with `grade()` function
