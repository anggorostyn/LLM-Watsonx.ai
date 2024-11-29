#  -----------------------------------------------------------------------------------------
#  (C) Copyright IBM Corp. 2024 .
#  https://opensource.org/licenses/BSD-3-Clause
#  -----------------------------------------------------------------------------------------
import pytest

from ibm_watson_machine_learning.foundation_models.prompts import PromptTemplate
from ibm_watson_machine_learning.foundation_models.prompts import PromptTemplateManager
from ibm_watson_machine_learning.foundation_models.utils.enums import ModelTypes

"""
When adding some fixture here follow that pattern:
- first "little" fixtures, that  are returning somethin simple as `prompt_id`;
- second "more complex setups" like `fixture_setup_prompt_mgr`;
- last one "tear down methods"
"""


@pytest.fixture(scope='class', name="prompt_id")
def fixture_prompt_id(prompt_mgr):
    """
    Fixture that is getting prompt ID
        Args:
            prompt_mgr:

        return:
            prompt_id
    """
    prompt_id = prompt_mgr.store_prompt(PromptTemplate(name="My test prompt",
                                                       model_id=ModelTypes.FLAN_T5_XL,
                                                       input_text="What is a {object} and how does it work?",
                                                       input_variables=["object"])).prompt_id
    return prompt_id


@pytest.fixture(scope='class', name="model_id")
def fixture_model_id(prompt_mgr):
    """
    Fixture that is getting model ID
        Args:
            prompt_mgr:

        return:
            model_id
    """
    model_id = ModelTypes.STARCODER.value

    return model_id


@pytest.fixture(scope="class", name="prompt_mgr")
def fixture_setup_prompt_mgr(credentials, project_id):
    """
    Fixture that setup prompt template manager
        Args:
            credentials:
            project_id:

        return:
            Prompt object, after executing test runners goes back to fixture to delete cleanup prompts
    """
    prompt_mgr = PromptTemplateManager(credentials, project_id=project_id)
    return prompt_mgr


@pytest.fixture(scope='class', name="prompt_template")
def fixture_prompt_template(request, prompt_mgr):
    """
    Fixture that is creating template before test, and deleting it after test.
        Args:
            request:
            prompt_mgr:

        yield:
            Prompt object, after executing test runners goes back to fixture to delete cleanup prompts
    """
    prompt = prompt_mgr.store_prompt(PromptTemplate(name="My test prompt",
                                                    model_id=ModelTypes.FLAN_T5_XL,
                                                    input_text="What is a {object} and how does it work?",
                                                    input_variables=["object"]))
    yield prompt
    prompt_mgr.delete_prompt(prompt.prompt_id, force=True)
    prompt_mgr.delete_prompt(request.cls.attributes['prompt_id'], force=True)


@pytest.fixture(scope='function', name="prompt_template_teardown_method")
def fixture_prompt_template_teardown_method(request, prompt_mgr):
    """
    Fixture that it is called before testing class to proxy cleanup part
        Args:
            request:
            prompt_mgr:

        yield:
    """
    yield True
    if request.cls.stored_id is not None:
        prompt_mgr.delete_prompt(prompt_id=request.cls.stored_id, force=True)
    request.cls.stored_id = None


@pytest.fixture(scope='class', name="prompt_template_deployment_teardown_method")
def fixture_prompt_template_deployment_teardown_method(request, prompt_mgr):
    """
    Fixture that it is called before testing class to proxy cleanup part
        Args:
            request:
            prompt_mgr:

        yield:
            Prompt object, after executing test runners goes back to fixture to delete cleanup prompts
    """
    yield True
    if request.cls.stored_id is not None:
        prompt_mgr.delete_prompt(prompt_id=request.cls.prompt_id, force=True)
        print("Teardown")
    request.cls.stored_id = None
