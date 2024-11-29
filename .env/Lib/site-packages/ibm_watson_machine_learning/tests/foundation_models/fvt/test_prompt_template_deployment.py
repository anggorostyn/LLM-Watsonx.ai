#  -----------------------------------------------------------------------------------------
#  (C) Copyright IBM Corp. 2023-2024.
#  https://opensource.org/licenses/BSD-3-Clause
#  -----------------------------------------------------------------------------------------

import pytest
from ibm_watson_machine_learning.tests.utils import is_cp4d

from ibm_watson_machine_learning.foundation_models import ModelInference
from ibm_watson_machine_learning.foundation_models.utils.enums import ModelTypes
from ibm_watson_machine_learning.wml_client_error import WMLClientError


@pytest.mark.usefixtures('prompt_template_deployment_teardown_method')
@pytest.mark.skipif(is_cp4d(), reason="Not supported on CP4D")
class TestPromptTemplateDeployment:
    """
    The test can be run on Cloud only.
    This test runs e2e Prompt Template deployments 
    """

    stored_id = None

    def test_00_create_deployment(self, api_client, prompt_mgr, prompt_id, project_id):
        api_client.set.default_project(project_id)

        TestPromptTemplateDeployment.base_model_id = ModelTypes.FLAN_T5_XL.value
        meta_props = {
            api_client.deployments.ConfigurationMetaNames.NAME: "SAMPLE DEPLOYMENT PROMPT TEMPLATE",
            api_client.deployments.ConfigurationMetaNames.ONLINE: {},
            api_client.deployments.ConfigurationMetaNames.BASE_MODEL_ID: TestPromptTemplateDeployment.base_model_id}

        deployment_details = api_client.deployments.create(prompt_id, meta_props)
        TestPromptTemplateDeployment.deployment_id = api_client.deployments.get_uid(deployment_details)

        assert isinstance(TestPromptTemplateDeployment.deployment_id, str), "`deployment_id` it is not String instance"

    def test_00b_create_deployment_without_base_model_id(self, api_client, prompt_id, prompt_mgr):

        meta_props = {
            api_client.deployments.ConfigurationMetaNames.NAME: "SAMPLE DEPLOYMENT PROMPT TEMPLATE",
            api_client.deployments.ConfigurationMetaNames.ONLINE: {}
        }

        deployment_details = api_client.deployments.create(prompt_id, meta_props)
        TestPromptTemplateDeployment.deployment_id_without_model = api_client.deployments.get_uid(deployment_details)

        assert api_client.deployments.get_details(TestPromptTemplateDeployment.deployment_id_without_model) \
                   .get('entity', {}).get('base_model_id') == prompt_mgr.load_prompt(prompt_id).model_id, "Taken `base_model_id` it is not equal to `model_id`"

    def test_00c_create_deployment_without_project_space(self, api_client, prompt_id, credentials, project_id):
        if credentials.get('project_id'):
            credentials.pop('project_id')
        if credentials.get('space_id'):
            credentials.pop('space_id')

        meta_props = {
            api_client.deployments.ConfigurationMetaNames.NAME: "SAMPLE DEPLOYMENT PROMPT TEMPLATE",
            api_client.deployments.ConfigurationMetaNames.ONLINE: {},
            api_client.deployments.ConfigurationMetaNames.BASE_MODEL_ID: TestPromptTemplateDeployment.base_model_id}

        deployment_details = api_client.deployments.create(prompt_id, meta_props)
        TestPromptTemplateDeployment.deployment_id = api_client.deployments.get_uid(deployment_details)

        assert pytest.raises(WMLClientError, api_client.deployments.create, project_id,
                             meta_props)

    def test_01_get_details(self, api_client, prompt_id):
        details = api_client.deployments.get_details(TestPromptTemplateDeployment.deployment_id)

        assert (details.get('entity', {}).get('prompt_template', {}).get('id', "")) == prompt_id
        assert (details.get('entity', {}).get('base_model_id', "")) == TestPromptTemplateDeployment.base_model_id

    def test_02_deployment_list(self, api_client):
        df = api_client.deployments.list()
        df_prompt = df[(df['GUID'] == TestPromptTemplateDeployment.deployment_id)]

        assert df_prompt.iloc[0]['ARTIFACT_TYPE'] == 'foundation_model', 'Wrong `ARTIFACT_TYPE` of asset'

    def test_03_generate_without_prompt_variables(self, api_client):
        with pytest.raises(WMLClientError):
            api_client.deployments.generate(TestPromptTemplateDeployment.deployment_id)

    def test_04_generate(self, api_client):
        generate_response = api_client.deployments.generate(TestPromptTemplateDeployment.deployment_id,
                                                            params={"prompt_variables": {"object": "loan"}})
        assert isinstance(generate_response, dict)
        assert generate_response.get('model_id', "") == TestPromptTemplateDeployment.base_model_id
        assert isinstance(generate_response.get('results', [{}])[0].get('generated_text'), str), \
            'Generated text it is not a `String Type`!'

    def test_05_generate_text(self, api_client):
        generated_text = api_client.deployments.generate_text(TestPromptTemplateDeployment.deployment_id,
                                                              params={"prompt_variables": {"object": "loan"}})

        assert isinstance(generated_text, str), 'Generated text it is not a `String Type`!'

    def test_06_generate_stream_text(self, api_client):
        generated_text_stream = \
            list(api_client.deployments.generate_text_stream(TestPromptTemplateDeployment.deployment_id,
                                                             params={"prompt_variables": {
                                                                 "object": "loan"}}))[0]

        assert isinstance(generated_text_stream, str), 'Text stream it is not a `String Type`!'

    def test_07_model_generate_without_variables(self, api_client):
        model = ModelInference(deployment_id=TestPromptTemplateDeployment.deployment_id,
                               api_client=api_client)

        with pytest.raises(WMLClientError):
            model.generate(TestPromptTemplateDeployment.deployment_id)

    def test_08_model_generate(self, api_client):
        model = ModelInference(deployment_id=TestPromptTemplateDeployment.deployment_id,
                               api_client=api_client)

        generate_response = model.generate(params={"prompt_variables": {"object": "loan"}})

        assert isinstance(generate_response, dict), 'Generated response it is not a `Dict Type`!'
        assert generate_response.get('model_id', "") == TestPromptTemplateDeployment.base_model_id, \
            'Generated response model it is not equal to `base_model_id`!'
        assert isinstance(generate_response.get('results', [{}])[0].get('generated_text'), str), \
            'Generated response it is not a `String Type`!'

    def test_09_model_generate_text(self, api_client):
        model = ModelInference(deployment_id=TestPromptTemplateDeployment.deployment_id,
                               api_client=api_client)
        assert isinstance(model.generate_text(params={"prompt_variables": {"object": "loan"}}), str), \
            "'Generated text model it is not a `String Type`!'"

    def test_10_model_generate_stream_text(self, api_client):
        model = ModelInference(deployment_id=TestPromptTemplateDeployment.deployment_id,
                               api_client=api_client)
        assert isinstance(list(model.generate_text_stream(params={"prompt_variables": {"object": "loan"}}))[0], str), \
            "'Generated model stream text it is not a `String Type`!'"

    def test_11_model_credentials_generate(self, credentials, project_id):
        model = ModelInference(deployment_id=TestPromptTemplateDeployment.deployment_id,
                               credentials=credentials, project_id=project_id)
        assert isinstance(list(model.generate_text_stream(params={"prompt_variables": {"object": "loan"}}))[0], str), \
            "Generated model credentials it is not `String Type`!"

    def test_12_update(self, api_client):
        new_name = "Changed Name"
        metadata = api_client.deployments.update(TestPromptTemplateDeployment.deployment_id,
                                                 changes={api_client.deployments.ConfigurationMetaNames.NAME: new_name})
        assert metadata.get('entity', {}).get('name', "") == new_name, "Update has not been applied"

    def test_13_delete_deployment(self, api_client):
        delete_deployment = api_client.deployments.delete(TestPromptTemplateDeployment.deployment_id)
        assert delete_deployment == "SUCCESS"
