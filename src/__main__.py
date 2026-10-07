import json
from typing import Dict, Any, List
from pydantic import BaseModel, ValidationError
import sys
import argparse
import numpy as np
from pathlib import Path
from llm_sdk import Small_LLM_Model


class FunctionDef(BaseModel):
    name: str
    description: str
    parameters: Dict[str, Any]
    returns: Dict[str, Any]


class UserPrompt(BaseModel):
    prompt: str


class FinalAnswer(BaseModel):
    prompt: str
    name: str
    parameters: Dict[str, Any]


def read_file(filepath: Path) -> str:
    try:
        with open(filepath, "r", encoding="utf-8") as f:
            return f.read()
    except FileNotFoundError:
        print(f"Error:No file founded in path '{filepath}'.")
        sys.exit(1)
    except PermissionError:
        print(f"Error: No permission for reading the file '{filepath}'.")
        sys.exit(1)
    except Exception as e:
        print(f"Unexpected error at reading '{filepath}': {e}")
        sys.exit(1)


def parse_prompts(json_text: str) -> List[UserPrompt]:
    try:
        raw_prompts = json.loads(json_text)
        validated_prompts = []

        for p_dict in raw_prompts:
            prompt_obj = UserPrompt(**p_dict)
            validated_prompts.append(prompt_obj)

        return validated_prompts

    except json.JSONDecodeError:
        print("Error: JSON format in the prompt is not valid")
        return []
    except ValidationError as e:
        print("Validation error: there is no 'prompt' or is not a text.")
        print(e)
        return []


def load_functions(json_text: str) -> List[FunctionDef]:
    try:
        raw_functions = json.loads(json_text)

        validated_functions = []

        for fn_dict in raw_functions:
            func_obj = FunctionDef(**fn_dict)
            validated_functions.append(func_obj)

        return validated_functions

    except json.JSONDecodeError:
        print("Error:JSON format is not valid.")
        return []
    except ValidationError:
        print("Validation error: JSON doesnt have "
              "the expected structure")
        return []


def build_system_prompt(functions: List[FunctionDef]) -> str:
    prompt = (
        "You are an AI assistant that must choose the correct function "
        "to call based on the user's request.\n\n"
        "Here are the available functions:\n"
    )

    for fn in functions:
        prompt += f"Function Name: {fn.name}\n"
        prompt += f"Description: {fn.description}\n"
        prompt += f"Parameters Schema: {json.dumps(fn.parameters)}\n"

    prompt += (
        "\nINSTRUCTIONS:\n"
        "1. You must respond ONLY with a valid JSON object.\n"
        "2. Do not add any extra text, explanations, or markdown formatting.\n"
        "3. The JSON must follow this exact structure:\n"
        '   {"name": "<function_name>", "parameters": {<key_value_pairs>}}\n'
        "4. PRESERVE ALL QUOTES AND PUNCTUATION. If the user"
        "  uses double quotes (\") or backslashes (\\), keep them and escape"
        "  them as \\\" and \\\\.\n\n"
        "EXAMPLES:\n"
        "User request: Format template: Say \"yes\" to {user}\n"
        '{"name": "fn_format_template", "parameters": '
        '{"template": "Say \\"yes\\" to {user}"}}\n\n'
        "User request: Read C:\\test\\file.txt\n"
        '{"name": "fn_read_file", "parameters": '
        '{"path": "C:\\\\test\\\\file.txt"}}\n'
    )

    return prompt


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Call Me Maybe - Constrained Function Calling"
    )

    parser.add_argument(
        "--functions_definition",
        type=Path,
        default=Path("data/input/functions_definition.json"),
        help="Route to the file with the definitions of the funcs"
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("data/input/function_calling_tests.json"),
        help="Route to the file with the prompts"
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/output/function_calls.json"),
        help="Route were the output will be saved"
    )
    return parser.parse_args()


def generate_function_call(model: Small_LLM_Model, system_prompt: str,
                           user_prompt: str,
                           max_tokens: int = 150) -> str:
    full_prompt = (
        f"<|im_start|>system\n{system_prompt}<|im_end|>\n"
        f"<|im_start|>user\n{user_prompt}<|im_end|>\n"
        f'<|im_start|>assistant\n{{"name": "')

    input_tensor = model.encode(full_prompt)
    input_ids = input_tensor.tolist()[0]
    end_token_tensor = model.encode("<|im_end|>")
    end_token_id = end_token_tensor.tolist()[0][0]
    generated_ids = []
    brackets_open = 1

    for step in range(max_tokens):
        logits = np.array(model.get_logits_from_input_ids(input_ids))
        next_token_id = int(np.argmax(logits))

        token_text = model.decode([next_token_id])
        if "{" in token_text:
            brackets_open += token_text.count("{")
        if "}" in token_text:
            brackets_open -= token_text.count("}")

        generated_ids.append(next_token_id)
        input_ids.append(next_token_id)

        if brackets_open <= 0:
            break

        if next_token_id == end_token_id:
            break

    return '{"name": "' + model.decode(generated_ids)


def fix_parameters(params: dict[Any, Any], target_fn: Any) -> dict[Any, Any]:
    if not isinstance(params, dict):
        return params

    schema_props = {}
    if target_fn and hasattr(target_fn, "parameters"):
        schema_props = target_fn.parameters

    fixed_params: dict[str, Any] = {}
    for key, val in params.items():
        expected_type = schema_props.get(key, {}).get("type")
        try:
            if expected_type == "number":
                fixed_params[key] = float(val)
            elif expected_type == "integer":
                fixed_params[key] = int(val)
            elif expected_type == "string":
                fixed_params[key] = str(val)
            else:
                fixed_params[key] = val
        except (ValueError, TypeError):
            fixed_params[key] = val

    return fixed_params


def main() -> None:
    args = parse_arguments()

    functions_content = read_file(args.functions_definition)
    prompts_content = read_file(args.input)

    functions = load_functions(functions_content)
    prompts = parse_prompts(prompts_content)

    if not functions or not prompts:
        print("Error: Not valid data, check the JSON")
        sys.exit(1)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    system_prompt = build_system_prompt(functions)

    print("Loading Qwen/Qwen3-0.6B...")
    model = Small_LLM_Model()

    results = []

    for idx, test_case in enumerate(prompts, start=1):
        print(f"\n--- Case  #{idx} ---")
        print(f"Prompt: {test_case.prompt}")
        raw_output = generate_function_call(model, system_prompt,
                                            test_case.prompt)
        raw_output = raw_output.replace("\\'", "'")
        try:
            parsed_json = json.loads(raw_output.strip())

            func_name = parsed_json.get("name", "unknown_function")
            raw_params = parsed_json.get("parameters", {})

            target_fn = next((f for f in functions if f.name == func_name),
                             None)

            corrected_params = fix_parameters(raw_params, target_fn)
            final_obj = FinalAnswer(
                prompt=test_case.prompt,
                name=func_name,
                parameters=corrected_params
            )
            results.append(final_obj.model_dump())
            print(f"Awnser :\n{final_obj}\n")

        except json.JSONDecodeError as e:
            print(f"Error: The model fail creating a valid JSON, cause: {e}")
            results.append({
                "prompt": test_case.prompt,
                "name": "error_parsing_json",
                "parameters": {}
            })
        except ValidationError as e:
            print(f"Error de validación Pydantic: {e}")
            results.append({
                "prompt": test_case.prompt,
                "name": "error_validation",
                "parameters": {}
            })

    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=4, ensure_ascii=False)

    print(f"Awnsers saved at {args.output}")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        sys.exit(0)
