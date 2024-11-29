#  -----------------------------------------------------------------------------------------
#  (C) Copyright IBM Corp. 2024.
#  https://opensource.org/licenses/BSD-3-Clause
#  -----------------------------------------------------------------------------------------

from ibm_watson_machine_learning.foundation_models.utils.enums import ModelTypes, PromptTuningInitMethods


def test_prompt_tuning_c(feature_step, universal_step, data_storage):
    prompt_tuning_info_c = dict(
        name="SDK test Classification Container",
        task_id="classification",
        base_model=ModelTypes.LLAMA_2_13B_CHAT,
        init_method=PromptTuningInitMethods.TEXT,
        init_text='text',
        num_epochs=2
    )

    universal_step.space_cleanup_qa()
    feature_step.write_data_to_container()
    universal_step.initialize_tune_experiment()
    feature_step.data_reference_setup_c()
    feature_step.read_saved_remote_data_before_fit()
    universal_step.initialize_prompt_tuner(prompt_tuning_info_c)
    universal_step.get_configuration_parameters_of_prompt_tuner()
    universal_step.run_prompt_tuning()
    universal_step.get_train_data()
    universal_step.get_run_status_prompt()
    universal_step.get_run_details_prompt()
    universal_step.get_run_details_include_metrics()
    universal_step.get_tuner()
    universal_step.list_all_runs()
    universal_step.list_specific_runs()
    universal_step.runs_get_last_run_details()
    universal_step.runs_get_specific_run_details()
    universal_step.runs_get_run_details_include_metrics()
    universal_step.get_summary_details()
    universal_step.store_prompt_tuned_model_default_params()
    universal_step.promote_model_to_deployment_space()
    universal_step.get_model_details()
    universal_step.list_repository()
    feature_step.deployment_creation_in_project()
    universal_step.response_from_deployment_inference()
    feature_step.delete_deployments()
    universal_step.delete_experiment()
    universal_step.delete_models()
    feature_step.delete_container()


def test_prompt_tuning_ca(feature_step, universal_step, data_storage):
    prompt_tuning_info_ca_default = dict(
        name="SDK test Classification with COS Connected Asset",
        task_id="classification",
        base_model=ModelTypes.GRANITE_13B_INSTRUCT_V2,
    )
    universal_step.space_cleanup_qa()
    feature_step.prepare_COS_instance_and_connection()
    universal_step.initialize_tune_experiment()
    feature_step.data_reference_setup_ca_default()
    feature_step.read_saved_remote_data_before_fit()
    universal_step.initialize_prompt_tuner(prompt_tuning_info_ca_default)
    universal_step.get_configuration_parameters_of_prompt_tuner()
    universal_step.run_prompt_tuning()
    universal_step.get_train_data()
    universal_step.get_run_status_prompt()
    universal_step.get_run_details_prompt()
    universal_step.get_run_details_include_metrics()
    universal_step.get_tuner()
    universal_step.list_all_runs()
    universal_step.list_specific_runs()
    universal_step.runs_get_last_run_details()
    universal_step.runs_get_specific_run_details()
    universal_step.runs_get_run_details_include_metrics()
    universal_step.get_summary_details()
    universal_step.store_prompt_tuned_model_default_params()
    universal_step.promote_model_to_deployment_space()
    universal_step.get_model_details()
    universal_step.list_repository()
    feature_step.read_results_reference_filename()
    universal_step.response_from_deployment_inference()
    universal_step.delete_experiment()
    universal_step.delete_models()
    feature_step.delete_connection_and_connected_data_asset()


def test_prompt_tuning_da(feature_step, universal_step, data_storage):
    prompt_tuning_info_da_default = dict(
        name="SDK test Classification",
        task_id="classification",
        base_model='google/flan-t5-xl',
        # accumulate_steps=2,
        # batch_size=16,
        # init_method=PromptTuningInitMethods.TEXT,
        # init_text="text",
        # learning_rate=0.1,
        # max_input_tokens=256,
        # max_output_tokens=128,
        # num_epochs=20,
        # tuning_type=PromptTuningTypes.PT,
        # verbalizer='Input: {{input}} Output:',
        # auto_update_model=False
    )

    universal_step.space_cleanup_qa()
    feature_step.prepare_data_asset()
    universal_step.initialize_tune_experiment()
    feature_step.data_reference_setup_da()
    feature_step.read_saved_remote_data_before_fit()
    universal_step.initialize_prompt_tuner(prompt_tuning_info_da_default)
    universal_step.get_configuration_parameters_of_prompt_tuner()
    universal_step.run_prompt_tuning()
    universal_step.get_train_data()
    universal_step.get_run_status_prompt()
    universal_step.get_run_details_prompt()
    universal_step.get_run_details_include_metrics()
    universal_step.get_tuner()
    universal_step.list_all_runs()
    universal_step.list_specific_runs()
    universal_step.runs_get_last_run_details()
    universal_step.runs_get_specific_run_details()
    universal_step.runs_get_run_details_include_metrics()
    universal_step.get_summary_details()
    universal_step.store_prompt_tuned_model_default_params()
    universal_step.promote_model_to_deployment_space()
    universal_step.get_model_details()
    universal_step.list_repository()
    feature_step.deployment_creation_with_prompted_asset()
    universal_step.response_from_deployment_inference()
    feature_step.delete_deployments_da()
    universal_step.delete_experiment()
    universal_step.delete_models()
    feature_step.delete_data_asset()
