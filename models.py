from typing import Callable
import requests
import json

API_KEYS = {
    "baseten": "DZ0KPsrZ.dGcR1CXBTSQSTzH1Vur2GCX5W1kSk7PL",
    "hugging_face": "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",
}

MODEL_IDS = {
    "falcon": {"id": "1PylM20", "version": "qeld103"},
    "wizard": {"id": "DBODNA0", "version": "w67yzgq"},
}

MODEL_SETTINGS = {
    "falcon": {
        "prompt_keys": ["prompt", "max_new_tokens", "return_full_text"],
        "default_values": {"return_full_text": False},
    },
    "wizard": {
        "prompt_keys": ["prompt", "max_new_tokens", "return_full_text"],
        "default_values": {"return_full_text": False},
    },
}

BASETEN_API_URL = "https://app.baseten.co"


class APIError(Exception):
    def __init__(self, message):
        self.message = message
        super().__init__(self.message)


# post request header
def create_auth_header(service: str) -> dict:
    match service:
        case "baseten":
            return {"Authorization": f"Api-Key {API_KEYS[service]}"}
        case "hugging_face":
            return {"Authorization": f"Bearer {API_KEYS[service]}"}
        case _:
            return {}


# post request header
def create_content_header(service: str) -> dict:
    match service:
        case "baseten":
            return {
                "content-type": "application/json",
                "Authorization": f"Api-Key {API_KEYS[service]}",
            }
        case _:
            return {}


# post request data
def build_prompt_payload(model_name: str, prompt: str, max_tokens: int) -> str:
    model_inputs: dict = {
        key: MODEL_SETTINGS[model_name]["default_values"].get(
            key, prompt if key == "prompt" else max_tokens
        )
        for key in MODEL_SETTINGS[model_name]["prompt_keys"]
    }
    return json.dumps(model_inputs)


# constructs predict for predictions
def construct_predict_url(model_name: str) -> str:
    return f"{BASETEN_API_URL}/models/{MODEL_IDS[model_name]['id']}/predict"


# constructs graphql for queries
def construct_graphql_url(service: str) -> str:
    return f"{BASETEN_API_URL}/graphql/"


# falcon model
def falcon(response: dict) -> str:
    return response["model_output"]["data"]["generated_text"]


# wizard model
def wizard(response: dict) -> str:
    return response["model_output"]


# match name to model function
MODEL_PROCESSORS = {
    "falcon": falcon,
    "wizard": wizard,
    # ... add other model processor functions here ...
}


# formats the response specific to each model
def process_response(model_name: str, response: dict):
    model_function: Callable = MODEL_PROCESSORS[model_name]

    if not model_function:
        raise APIError(f"Unknown model: {model_name}")

    try:
        return model_function(response)
    except Exception as e:
        raise APIError(f"Failed to process response: {e}")


# runs specified model with given prompt
def run_model(prompt: str, model_name: str, max_tokens: int) -> str:
    json_input: str = build_prompt_payload(model_name, prompt, max_tokens)

    try:
        response: dict = requests.post(
            construct_predict_url(model_name),
            headers=create_content_header("baseten"),
            data=json_input,
        ).json()

        if response is None:
            raise APIError("NoneType")
        if "error" in response:
            raise APIError(response["error"])

    except Exception as e:
        print("Error: ", e)
        raise APIError(str(e))

    return process_response(model_name, response)


# activates or deactivates model
def set_model_status(model_name: str, desired_status: bool) -> bool:
    operation: str = "deactivate" if not desired_status else "activate"

    mutation_query: str = f"""
    mutation {{
        {operation}_model_version(model_version_id: "{MODEL_IDS[model_name]['version']}") {{
          ok
        }}
    }}
    """

    response: dict = requests.post(
        construct_graphql_url("baseten"),
        data={"query": mutation_query},
        headers=create_auth_header("baseten"),
    ).json()

    return "errors" not in response


if __name__ == "__main__":
    pass
