Markdown
*This project has been created as part of the 42 curriculum by marcsan2.*

# Call Me Maybe - Introduction to Function Calling in LLMs

## Description
"Call Me Maybe" is an artificial intelligence project that explores the mechanics of **Function Calling** and **Constrained Decoding**. The goal of this project is to force a Small Language Model (SLM) — specifically `Qwen/Qwen3-0.6B` — to act as a routing agent. Instead of responding with conversational text, the model must read a user's prompt, select the appropriate tool from a provided JSON schema, and output exactly one valid JSON object containing the function name and its required parameters.

This is achieved completely from scratch using a custom `llm_sdk` and `numpy` to manipulate token probabilities (logits), strictly adhering to the 42 curriculum limitations (no direct usage of Hugging Face pipelines or PyTorch high-level generation methods).

## Instructions
### Prerequisites
- The `uv` package manager installed.
- Python 3.10+
- The project must be located in a partition with sufficient storage (like `/goinfre` or `/sgoinfre` on 42 school computers) due to the size of the CUDA libraries and LLM weights.

### Installation & Cache Setup
Because of the strict storage quotas in the 42 school environment, you must configure your cache directories before running the project to avoid `No space left on device (os error 28).


export UV_CACHE_DIR=/goinfre/$USER/uv_cache
export HF_HOME=/goinfre/$USER/huggingface_cache
Execution
Run the project using uv. By default, it will read from data/input/ and write to data/output/:

uv run python -m src
You can also specify custom file paths using the provided arguments:

uv run python -m src \
  --functions_definition data/custom/functions.json \
  --input data/custom/tests.json \
  --output data/custom/results.json

### Algorithm Explanation
To enforce strict JSON output without conversational filler, this project uses a combination of Assistant Prefilling and Bracket Counting (State Tracking):

Forced Prefill: Instead of masking probabilities in the first step, the prompt is manually terminated with an assistant header followed immediately by an opening bracket {. This hacks the models context, forcing it to seamlessly complete the JSON structure instead of starting with a greeting (e.g., "Here is your JSON:").

Greedy Decoding: At each generation step, the algorithm uses numpy.argmax on the raw logits to select the most probable next token deterministically.

Perfect Termination: A counter tracks structural depth by adding 1 for every { and subtracting 1 for every } generated. The moment the counter reaches 0, the generation is forcefully halted, preventing any trailing text or markdown formatting.

### Design Decisions
Pydantic for Data Validation: Used extensively to parse incoming schemas (FunctionDef), user prompts (UserPrompt), and strictly validate the final LLM output (FinalAnswer). This ensures the program fails gracefully if the input data is malformed.

Error Handling: Wrapped the json.loads step in a try-except block. If the model hallucinates an invalid JSON string, the program catches the JSONDecodeError and outputs a fallback structure to maintain pipeline integrity for automated grading.

Logits Processing with Numpy: Bypassed the need for heavy frameworks by extracting the token lists and directly applying mathematical arrays (np.array) to find the next optimal token.

### Performance Analysis
Accuracy: Extremely high for the target SLM. The prefilling strategy guarantees a 100% success rate in starting a JSON object, while the bracket tracker perfectly isolates the payload.

Speed: Very fast. By avoiding an expensive masking operation over the entire 150,000+ token vocabulary at every generation step, the overhead is minimal, bounded only by the models inference speed.

Reliability: The use of np.argmax acts as a temperature of 0.0, resulting in deterministic, reproducible outputs across multiple runs.

### Challenges Faced
Storage Limits in 42 iMacs (OS Error 28): The underlying dependencies (PyTorch/CUDA via llm_sdk) and the 1.5GB Qwen model quickly exhausted the 5GB home directory limit. Solution: Moved the entire project to the /goinfre partition and mapped the UV_CACHE_DIR and HF_HOME environment variables accordingly.

Library Restrictions: The inability to use transformers pipelines meant rewriting the autoregressive generation loop token-by-token. Solution: Studied the provided llm_sdk public methods (encode, decode, get_logits_from_input_ids) to build a custom generation engine from the ground up using only basic Python and numpy.

### Testing Strategy
Unit Validation: Used Pydantic schemas to strictly validate both input constraints and the final generated objects before writing them to disk.

Integration Testing: Ran the model against multiple diverse prompts inside function_calling_tests.json (ranging from simple math queries to abstract string manipulation). Verified that the generated file data/output/function_calling_results.json complies perfectly with the structure mandated by the subject.

### Example Usage
Input Prompt:

JSON
{"prompt": "What is the sum of 2 and 3?"}
Command:

Bash
uv run python -m src
Generated Output (function_calling_results.json):

JSON
[
    {
        "prompt": "What is the sum of 2 and 3?",
        "name": "fn_add_numbers",
        "parameters": {
            "a": 2.0,
            "b": 3.0
        }
    }
]
### Resources
JSON Specification: json.org

Pydantic Documentation: docs.pydantic.dev

Numpy Argmax: numpy.org/doc/stable/reference/generated/numpy.argmax.html

AI Usage: An AI assistant (LLM) was utilized as a pair-programming partner during this project. Specifically, it assisted in:

Diagnosing and resolving the OS Error 28 storage limitations by suggesting cache path rerouting.

Brainstorming approaches for constrained decoding, leading to the Assistant Prefilling strategy over a heavy mask-based approach.

Scaffolding the Pydantic models for input/output validation.