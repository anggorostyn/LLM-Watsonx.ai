#  -----------------------------------------------------------------------------------------
#  (C) Copyright IBM Corp. 2024.
#  https://opensource.org/licenses/BSD-3-Clause
#  -----------------------------------------------------------------------------------------

import json

import pytest
from pandas import DataFrame

from ibm_watson_machine_learning.experiment import TuneExperiment
from ibm_watson_machine_learning.foundation_models.inference import ModelInference
from ibm_watson_machine_learning.foundation_models.prompt_tuner import PromptTuner
from ibm_watson_machine_learning.tests.utils import get_space_id
from ibm_watson_machine_learning.tests.utils.cleanup import space_cleanup
from ibm_watson_machine_learning.wml_client_error import ApiRequestFailure, WMLClientError


class UniversalSteps:
    def __init__(self, data_storage):
        self.data_storage = data_storage

    def space_cleanup_qa(self):
        space_checked = False
        while not space_checked:
            space_cleanup(self.data_storage.api_client,
                          get_space_id(self.data_storage.api_client, self.data_storage.space_name,
                                       cos_resource_instance_id=self.data_storage.cos_resource_instance_id),
                          days_old=7)
            space_id = get_space_id(self.data_storage.api_client, self.data_storage.space_name,
                                    cos_resource_instance_id=self.data_storage.cos_resource_instance_id)
            try:
                assert space_id is not None, "space_id is None"
                self.data_storage.api_client.spaces.get_details(space_id)
                space_checked = True
            except AssertionError or ApiRequestFailure:
                space_checked = False

        self.data_storage.space_id = space_id

        if self.data_storage.SPACE_ONLY:
            self.data_storage.api_client.set.default_space(self.data_storage.space_id)
        else:
            self.data_storage.api_client.set.default_project(self.data_storage.project_id)

    def initialize_tune_experiment(self):
        if self.data_storage.SPACE_ONLY:
            self.data_storage.experiment = TuneExperiment(credentials=self.data_storage.credentials.copy(),
                                                          space_id=self.data_storage.space_id)
        else:
            self.data_storage.experiment = TuneExperiment(credentials=self.data_storage.credentials.copy(),
                                                          project_id=self.data_storage.project_id)

        assert isinstance(self.data_storage.experiment, TuneExperiment), "Experiment is not of type TuneExperiment."

    def initialize_prompt_tuner(self, prompt_tuning_data_set):
        self.data_storage.prompt_tuner = self.data_storage.experiment.prompt_tuner(**prompt_tuning_data_set)

        assert isinstance(self.data_storage.prompt_tuner,
                          PromptTuner), "experiment.prompt_tuner did not return PromptTuner object"

    def get_configuration_parameters_of_prompt_tuner(self):
        parameters = self.data_storage.prompt_tuner.get_params()
        print(parameters)

        assert isinstance(parameters, dict), 'Config parameters are not a dictionary instance.'

    def run_prompt_tuning(self):
        self.data_storage.tuned_details = self.data_storage.prompt_tuner.run(
            training_data_references=self.data_storage.train_data_connections,
            training_results_reference=self.data_storage.results_data_connection,
            background_mode=False)

        self.data_storage.run_id = self.data_storage.tuned_details['metadata']['id']

    def get_train_data(self):
        binary_data = self.data_storage.prompt_tuner.get_data_connections()[0].read(binary=True)
        try:
            self.data_storage.train_data = json.loads(binary_data.decode())
        except json.decoder.JSONDecodeError:
            self.data_storage.train_data = [
                json.loads(line) for line in binary_data.decode().splitlines() if line
            ]

        print("Train data sample: \n")
        print(self.data_storage.train_data)

        assert isinstance(self.data_storage.train_data, list), "Trained Data it is not a list"
        assert len(self.data_storage.train_data) > 0, "Train Data is empty"

    def get_run_status_prompt(self):
        status = self.data_storage.prompt_tuner.get_run_status()
        run_details = self.data_storage.prompt_tuner.get_run_details()

        assert status == run_details['entity'].get('status', {}).get(
            'state'), "Different statuses returned. Status: {},\n\n Run details {}".format(status,
                                                                                           run_details)
        assert status == "completed", "Prompt Tuning run didn't finished successfully. Status: {},\n\n Run details {}" \
            .format(status, run_details)

    def get_run_details_prompt(self):
        parameters = self.data_storage.prompt_tuner.get_run_details()
        training_details = self.data_storage.api_client.training.get_details(training_uid=parameters['metadata']['id'])
        print(json.dumps(training_details, indent=4))
        print(parameters)

        assert parameters is not None, "Parameters cannot be None"

    def get_run_details_include_metrics(self):
        parameters = self.data_storage.prompt_tuner.get_run_details(include_metrics=True)
        training_details = self.data_storage.api_client.training.get_details(training_uid=parameters['metadata']['id'])
        print(json.dumps(training_details, indent=4))
        print(parameters)

        assert parameters is not None, "Parameters cannot be None"
        assert 'metrics' in parameters['entity']['status'], "prompt_tuner.get_run_details did not return metrics"

    def get_tuner(self):
        parameters = self.data_storage.prompt_tuner.get_run_details()
        self.data_storage.prompt_tuner = self.data_storage.experiment.runs.get_tuner(
            run_id=parameters['metadata']['id'])
        print("Received tuner params:", self.data_storage.prompt_tuner.get_params())

        assert isinstance(self.data_storage.prompt_tuner,
                          PromptTuner), "experiment.get_tuner did not return PromptTuner object"
        status = self.data_storage.prompt_tuner.get_run_status()
        assert status is not None, "Status cannot be None"

    def list_all_runs(self):
        historical_tunings = self.data_storage.experiment.runs.list()
        print("All historical prompt tunings:")
        print(historical_tunings)

        assert isinstance(historical_tunings, DataFrame), "experiment.runs did not return DataFrame object"

    def list_specific_runs(self):
        parameters = self.data_storage.prompt_tuner.get_params()
        historical_tunings = self.data_storage.experiment.runs(filter=parameters['name']).list()
        print(f"Prompt tunings with name {parameters['name']}:")
        print(historical_tunings)

        assert isinstance(historical_tunings, DataFrame), "experiment.runs did not return DataFrame object"

    def runs_get_last_run_details(self):
        run_details = self.data_storage.experiment.runs.get_run_details()
        print("Last prompt tuning run details:")
        print(run_details)

        assert run_details is not None, "run_details cannot be None"
        assert isinstance(run_details, dict), "experiment.runs.get_run_details did not return dict object"

    def runs_get_specific_run_details(self):
        parameters = self.data_storage.prompt_tuner.get_run_details()
        run_details = self.data_storage.experiment.runs.get_run_details(run_id=parameters['metadata']['id'])
        print(f"Run {parameters['metadata']['id']} details:")
        print(run_details)

        assert run_details is not None, "run_details cannot be None"
        assert isinstance(run_details, dict), "experiment.runs.get_run_details did not return dict object"

    def runs_get_run_details_include_metrics(self):
        run_details = self.data_storage.experiment.runs.get_run_details(include_metrics=True)
        print("Last prompt tuning run details:")
        print(run_details)

        assert run_details is not None, "run_details cannot be None"
        assert 'metrics' in run_details['entity']['status'], "experiment.runs.get_run_details did not return metrics"

    def get_summary_details(self):
        run_summary_details = self.data_storage.prompt_tuner.summary()
        run_details = self.data_storage.prompt_tuner.get_run_details()
        print(f"Run {self.data_storage.prompt_tuner.id} summary details:")
        print(run_summary_details)

        assert run_summary_details is not None, "run_summary_details cannot be None"
        assert type(run_summary_details) is DataFrame

        assert run_summary_details.get('Enhancements')[0][0] == run_details['entity'].get('prompt_tuning').get(
            'tuning_type'), ""
        assert run_summary_details.get('Base model')[0] == run_details['entity'].get('prompt_tuning').get('base_model')[
            'model_id'], ""
        assert run_summary_details.get('Auto store')[0] == run_details['entity'].get('auto_update_model'), ""
        assert run_summary_details.get('Epochs')[0] == run_details['entity'].get('prompt_tuning').get('num_epochs'), ""
        assert run_summary_details.get('loss')[0] > 0, "run_details cannot be Empty"

    def store_prompt_tuned_model_default_params(self):
        stored_model_details = self.data_storage.api_client.repository.store_model(
            training_id=self.data_storage.prompt_tuner.id)
        self.data_storage.stored_model_id = self.data_storage.api_client.repository.get_model_id(stored_model_details)

        assert self.data_storage.stored_model_id is not None, "stored_model_id cannot be None"

    def promote_model_to_deployment_space(self):
        if self.data_storage.SPACE_ONLY:
            self.data_storage.promoted_model_id = self.data_storage.stored_model_id  # no need to promote
        else:
            self.data_storage.promoted_model_id = self.data_storage.api_client.spaces.promote(
                self.data_storage.stored_model_id,
                source_project_id=self.data_storage.project_id,
                target_space_id=self.data_storage.space_id)
            self.data_storage.project_models_to_delete.append(self.data_storage.stored_model_id)

        self.data_storage.space_models_to_delete.append(self.data_storage.promoted_model_id)

    def get_model_details(self):
        print(self.data_storage.api_client.default_space_id)
        if self.data_storage.api_client.default_space_id is None:
            self.data_storage.api_client.set.default_space(self.data_storage.space_id)
        model_details = self.data_storage.api_client.repository.get_details(self.data_storage.promoted_model_id)

        assert model_details is not None, "model_details cannot be None"
        assert model_details['entity'].get('training_id') is not None, "training_id cannot be None"
        assert model_details['entity'].get('type') == 'prompt_tune_1.0', f"{model_details['entity'].get('type')} it is not equal to 'prompt_tune_1.0' "

    def list_repository(self):
        repository_data_frame_list = self.data_storage.api_client.repository.list()
        repository_list = repository_data_frame_list.get('GUID').to_list()

        assert self.data_storage.promoted_model_id in repository_list, "promoted_model_id it is not in repository_list"

    def response_from_deployment_inference(self):
        if hasattr(self, 'deployment_id'):
            deployment_id = self.data_storage.deployment_id
            d_inference = ModelInference(
                deployment_id=deployment_id,
                api_client=self.data_storage.api_client
            )
            response = d_inference.generate_text(
                prompt="sentence1: Oil prices fall back as Yukos oil threat lifted sentence2: Oil prices rise.")
            print(response)

            assert response is not None, "response cannot be None"

    def delete_experiment(self):
        self.data_storage.prompt_tuner.cancel_run(hard_delete=True)
        with pytest.raises(WMLClientError):
            self.data_storage.api_client.training.get_details(self.data_storage.run_id)

    def delete_models(self):
        for model in self.data_storage.space_models_to_delete:
            self.data_storage.api_client.repository.delete(model)

        if not self.data_storage.SPACE_ONLY:
            self.data_storage.api_client.set.default_project(self.data_storage.project_id)
            for model in self.data_storage.project_models_to_delete:
                self.data_storage.api_client.repository.delete(model)
