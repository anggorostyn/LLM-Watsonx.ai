#  -----------------------------------------------------------------------------------------
#  (C) Copyright IBM Corp. 2023-2024.
#  https://opensource.org/licenses/BSD-3-Clause
#  -----------------------------------------------------------------------------------------
from string import Formatter
from typing import Any, Sequence, List, Dict
from dataclasses import dataclass, KW_ONLY, asdict
from json import loads as json_loads
import warnings

from ibm_watson_machine_learning._wrappers import requests
from ibm_watson_machine_learning.helpers import DataConnection
from ibm_watson_machine_learning.messages.messages import Messages
from ibm_watson_machine_learning.wml_client_error import WMLClientError
from ibm_watson_machine_learning.utils.autoai.utils import load_file_from_file_system_nonautoai
from ibm_watson_machine_learning.utils.autoai.enums import DataConnectionTypes
from ibm_watson_machine_learning.lifecycle import SpecStates


@dataclass
class PromptTuningParams:
    base_model: dict
    _: KW_ONLY
    accumulate_steps: int = None
    batch_size: int = None
    init_method: str = None
    init_text: str = None
    learning_rate: float = None
    max_input_tokens: int = None
    max_output_tokens: int = None
    num_epochs: int = None
    task_id: str = None
    tuning_type: str = None
    verbalizer: str = None

    def to_dict(self):
        return {key: value for key, value in asdict(self).items() if value is not None}


def _get_foundation_models_spec(url, operation_name, additional_params: dict = None):
    params = {"version": "2023-09-30"}
    if additional_params:
        params.update(additional_params)
    response = requests.get(url,
                            params=params,
                            headers={'X-WML-User-Client': 'PythonClient'})
    if response.status_code == 200:
        return response.json()
    elif response.status_code == 404:
        raise WMLClientError(Messages.get_message(message_id="fm_prompt_tuning_no_foundation_models"))
    else:
        msg = f"{operation_name} failed. Reason: {response.text}"
        raise WMLClientError(msg)


def get_model_specs(url):
    try:
        return _get_foundation_models_spec(f"{url}/ml/v1-beta/foundation_model_specs",
                                           "Get available foundation models")
    except WMLClientError as e:
        raise WMLClientError(Messages.get_message(url, message_id="fm_prompt_tuning_no_model_specs"), e)


def get_model_lifecycle(url, model_id):
    if model_id is None:
        return None
    else:
        model_specs = get_model_specs(url=url)
        model_spec = next((model_metadata for model_metadata in model_specs.get('resources', []) 
                           if model_metadata.get('model_id') == model_id), None)
        return model_spec.get('lifecycle') if model_spec is not None else None


def _check_model_state(url, model_id):

    default_warning_template = (
        "Model '{model_id}' is in {state} state from {start_date} until {withdrawn_start_date}. "
        "IDs of alternative models: {alternative_model_ids}. "
        "Further details: https://dataplatform.cloud.ibm.com/docs/content/wsj/analyze-data/fm-model-lifecycle.html?context=wx&audience=wdp")

    lifecycle = get_model_lifecycle(url, model_id)

    modes_list = [ids.get("id") for ids in (lifecycle or [])]

    if lifecycle and SpecStates.DEPRECATED.value in modes_list and SpecStates.CONSTRICTED.value in modes_list:
        deprecated_and_constricted_warning_template = (
            "Model '{model_id}' is in deprecated state from {deprecated_start_date} until the model status is changed "
            "to constricted {constricted_start_date} state, which will last until {withdrawn_start_date}. "
            "IDs of alternative models: {alternative_model_ids}. "
            "Further details: https://dataplatform.cloud.ibm.com/docs/content/wsj/analyze-data/fm-model-lifecycle.html?context=wx&audience=wdp")
        warnings.warn(deprecated_and_constricted_warning_template.format(
            model_id=model_id,
            deprecated_start_date=next((el.get('start_date') for el in lifecycle if el.get('id') == SpecStates.DEPRECATED.value), None),
            constricted_start_date=next((el.get('start_date') for el in lifecycle if el.get('id') == SpecStates.CONSTRICTED.value), None),
            withdrawn_start_date=next((el.get('start_date') for el in lifecycle if el.get('id') == SpecStates.WITHDRAWN.value), None),
            alternative_model_ids=next((', '.join(el.get('alternative_model_ids', ["None"])) for el in lifecycle
                                        if el.get('id') == SpecStates.CONSTRICTED.value), None)
        ), category=LifecycleWarning)

    elif lifecycle and SpecStates.DEPRECATED.value in modes_list:
        warnings.warn(default_warning_template.format(
            model_id=model_id,
            state=SpecStates.DEPRECATED.value,
            start_date=next((el.get('start_date') for el in lifecycle if el.get('id') == SpecStates.DEPRECATED.value), None),
            withdrawn_start_date=next((el.get('start_date') for el in lifecycle if el.get('id') == SpecStates.WITHDRAWN.value), None),
            alternative_model_ids=next((', '.join(el.get('alternative_model_ids', ["None"])) for el in lifecycle
                                        if el.get('id') == SpecStates.DEPRECATED.value), None)
        ), category=LifecycleWarning)

    elif lifecycle and SpecStates.CONSTRICTED.value in modes_list:
        warnings.warn(default_warning_template.format(
            model_id=model_id,
            state=SpecStates.CONSTRICTED.value,
            start_date=next((el.get('start_date') for el in lifecycle if el.get('id') == SpecStates.CONSTRICTED.value), None),
            withdrawn_start_date=next((el.get('start_date') for el in lifecycle if el.get('id') == SpecStates.WITHDRAWN.value), None),
            alternative_model_ids=next((', '.join(el.get('alternative_model_ids', ["None"])) for el in lifecycle
                                        if el.get('id') == SpecStates.CONSTRICTED.value), None)
        ), category=LifecycleWarning)


def get_model_specs_with_prompt_tuning_support(url):
    try:
        return _get_foundation_models_spec(url=f"{url}/ml/v1-beta/foundation_model_specs",
                                           operation_name="Get available foundation models",
                                           additional_params={"filters": "function_prompt_tune_trainable"})
    except WMLClientError as e:
        raise WMLClientError(Messages.get_message(url, message_id="fm_prompt_tuning_no_model_specs"), e)


def get_supported_tasks(url) -> dict:
    try:
        return _get_foundation_models_spec(f"{url}/ml/v1-beta/foundation_model_tasks",
                                           "Get tasks that are supported by the foundation models.")
    except WMLClientError as e:
        raise WMLClientError(Messages.get_message(url, message_id="fm_prompt_tuning_no_supported_tasks"), e)


def get_all_supported_tasks_dict(url="https://us-south.ml.cloud.ibm.com") -> dict:
    tasks_dict = dict()
    for task_spec in get_supported_tasks(url).get('resources', []):
        tasks_dict[task_spec['label'].replace("-", "_").replace(" ", "_").upper()] = task_spec['task_id']
    return tasks_dict


def load_request_json(run_id, wml_client, run_params = None) -> dict:
    if run_params is None:
        run_params = wml_client.training.get_details(run_id)

    model_request_path = run_params['entity'].get('results_reference', {}).get('location', {}).get('model_request_path')

    if model_request_path is None:
        raise WMLClientError("Missing model_request_path in run_params. Verify if the training run has been completed.")

    if wml_client.CLOUD_PLATFORM_SPACES:
        results_reference = DataConnection._from_dict(run_params['entity']['results_reference'])

        if run_params['entity']['results_reference']['type'] == DataConnectionTypes.CA:
            results_reference.location.file_name = model_request_path
        else:
            results_reference.location.path = model_request_path

        results_reference.set_client(wml_client)
        request_json_bytes = results_reference.read(raw=True, binary=True)

        # download from cos

    elif wml_client.CPD_version >= 4.8:
        asset_parts = model_request_path.split('/')
        model_request_asset_url = '/'.join(asset_parts[asset_parts.index('assets') + 1:])
        request_json_bytes = load_file_from_file_system_nonautoai(wml_client=wml_client,
                                                                  file_path=model_request_asset_url).read()
    else:
        raise WMLClientError("Unsupported environment for this action")

    return json_loads(request_json_bytes.decode())


def is_training_prompt_tuning(training_id, wml_client):
    """Returns True if training_id is connected to prompt tuning"""
    if training_id is None:
        return False
    run_params = wml_client.training.get_details(training_uid=training_id)
    return bool(run_params['entity'].get('prompt_tuning'))


class TemplateFormatter(Formatter):
    def check_unused_args(self, 
                          used_args: List[(int | str)], 
                          args: Sequence, 
                          kwargs: Dict[str, Any]) -> None:
        """Check for unused args."""
        extra_args = set(kwargs).difference(used_args)
        if extra_args:
            raise KeyError(extra_args)


class HAPDetectionWarning(UserWarning): ...


class PIIDetectionWarning(UserWarning): ...


class LifecycleWarning(UserWarning): ...
