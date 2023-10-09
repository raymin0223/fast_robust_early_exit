import json
import os
import time
import traceback
import backoff
import openai
from datetime import datetime

DEFAULT_MODEL_IDS = [
    "text-davinci-003",
    "gpt-3.5-turbo",
    "gpt-3.5-turbo-0613",
    "gpt-3.5-turbo-0301",
    "text-curie-003"
]


@backoff.on_exception(backoff.expo, openai.error.RateLimitError)
def completions_with_backoff(**kwargs):
    return openai.Completion.create(**kwargs)


@backoff.on_exception(backoff.expo, openai.error.RateLimitError)
def chat_completions_with_backoff(**kwargs):
    return openai.ChatCompletion.create(**kwargs)


def log_openai_error(message: str, log_path: str):
    timestamp = datetime.now().astimezone().isoformat()
    with open(log_path, "a") as f:
        f.write(" {} ".format(timestamp).center(80, "#"))
        f.write("\n")
        f.write(message)
        f.write("\n")
        f.write("\n")


def get_openai_errors(log_path, lines=50):
    with open(log_path) as f:
        if lines > 0:
            return "".join(f.readlines()[-lines:])
        else:
            return f.read()


def create_completion(*args, model_type, log_path, verbose=True, error_while=None, **kwargs):
    retry_intervals = [0] * 1 + [1] * 5 + [10, 30, 60, 300]

    for i, t in enumerate(retry_intervals):
        if t:
            time.sleep(t)
        try:
            if model_type == "chat":
                response = openai.ChatCompletion.create(*args, **kwargs)
            elif model_type == "text":
                response = completions_with_backoff(*args, **kwargs)
            else:
                raise ValueError("Unknown Open AI model type: {}".format(model_type))
            return response
        except Exception as e:
            if verbose:
                print("Error during OpenAI completion attempt #{}: [{}] {}".format(i + 1, type(e).__name__, str(e)))
            if error_while is not None:
                log_openai_error("Error during {} attempt #{}:\n{}".format(error_while, i + 1, traceback.format_exc()), log_path)
            else:
                log_openai_error(traceback.format_exc(), log_path)
    else:
        return None


def get_model_id(model_id):
    if model_id in DEFAULT_MODEL_IDS:
        return model_id
    else:
        raise ValueError("Unknown model id: {}".format(model_id))