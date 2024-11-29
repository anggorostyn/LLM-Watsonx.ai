#  -----------------------------------------------------------------------------------------
#  (C) Copyright IBM Corp. 2023-2024.
#  https://opensource.org/licenses/BSD-3-Clause
#  -----------------------------------------------------------------------------------------
import pytest

from typing import Generator

from ibm_watson_machine_learning.tests.utils import get_wml_credentials, is_cp4d
from ibm_watson_machine_learning.foundation_models import Model
from ibm_watson_machine_learning.foundation_models.utils.enums import ModelTypes, DecodingMethods
from ibm_watson_machine_learning.metanames import GenTextParamsMetaNames as GenParams
from ibm_watson_machine_learning.metanames import GenTextReturnOptMetaNames as ReturnOpts
from ibm_watson_machine_learning.wml_client_error import WMLClientError
from ibm_watson_machine_learning.foundation_models.utils.utils import get_model_specs


wml_credentials = get_wml_credentials()
# model_types_list = [model.value for model in ModelTypes]
# available_models = [model_spec['model_id'] for model_spec in get_model_specs(wml_credentials.get('url')).get('resources', [])
#                     if model_spec['model_id'] in model_types_list]

#For automatic tests we select only one model
available_models = ['google/flan-ul2']


class TestTextGeneration:
    """
    This tests covers:
    - ...
    """
    wml_credentials = None
    project_id = None
    space_id = None

    @classmethod
    def setup_class(cls) -> None:
        """
        Load WML credentials from config.ini file based on ENV variable.
        """
        cls.wml_credentials = get_wml_credentials()
        cls.project_id = cls.wml_credentials.get('project_id')

    def test_01_create_flan_ul2_model(self):
        model_id = ModelTypes.FLAN_UL2
        ul2_params = {
            GenParams.DECODING_METHOD: DecodingMethods.GREEDY,
            GenParams.MAX_NEW_TOKENS: 50,
            GenParams.STOP_SEQUENCES: ['\n\n']
        }

        if is_cp4d():
            with pytest.raises(WMLClientError):
                _ = Model(model_id=model_id, params=ul2_params,
                          credentials=self.wml_credentials, project_id=self.project_id)

        else:
            flan_ul2_model = Model(
                model_id=model_id,
                params=ul2_params,
                credentials=self.wml_credentials,
                project_id=self.project_id)
            assert flan_ul2_model.get_details()['model_id'] == model_id.value, ("`model_id` from attribute and "
                                                                                "`get_details()` are not the same")

            q = "What is 1 + 1?"

            ul2_text = flan_ul2_model.generate_text(prompt=q)
            print(ul2_text)

            ul2_response = flan_ul2_model.generate(prompt=q)
            print(ul2_response['results'][0]['generated_text'])
            assert ul2_text == ul2_response['results'][0]['generated_text'], ("generated text from `generate()` and "
                                                                              "`generate_text()` are not the same")

            sample_q = "What is 1 + {}?"
            prompts = [sample_q.format(i) for i in range(20)]
            ul2_texts = flan_ul2_model.generate_text(prompt=prompts, concurrency_limit=5)
            assert len(prompts) == len(ul2_texts), "Length of texts list is not equal length of prompts list"

            ul2_responses = flan_ul2_model.generate(prompt=prompts, concurrency_limit=6)
            assert len(prompts) == len(ul2_responses), "Length of responses is not equal length of prompts list"

            for text, response in zip(ul2_texts, ul2_responses):
                assert text == response['results'][0]['generated_text'], \
                    f"Methods outputs are not equal generate_text: {text}, generate: {response}"

    def test_02_create_gpt_neox_model(self):
        model_id = ModelTypes.GPT_NEOX
        if is_cp4d():
            with pytest.raises(WMLClientError):
                _ = Model(model_id=model_id, credentials=self.wml_credentials, project_id=self.project_id)
        else:
            gpt_neox_model = Model(
                model_id=model_id,
                credentials=self.wml_credentials,
                project_id=self.project_id)
            assert gpt_neox_model.get_details()['model_id'] == model_id.value, ("`model_id` from attribute and"
                                                                                " `get_details()` are not the same")

            q = "What is 1 + 1?"

            text_params = {
                GenParams.DECODING_METHOD: DecodingMethods.GREEDY,
                GenParams.MAX_NEW_TOKENS: 10,
                GenParams.STOP_SEQUENCES: ["\n"]
            }
            neox_text = gpt_neox_model.generate_text(prompt=q, params=text_params)
            print(neox_text)

            text_params_updated = {
                GenParams.DECODING_METHOD: DecodingMethods.SAMPLE,
                GenParams.MAX_NEW_TOKENS: 10,
                GenParams.STOP_SEQUENCES: ["\n"],
                GenParams.TEMPERATURE: 0

            }

            neox_response = gpt_neox_model.generate(prompt=q, params=text_params_updated)
            print(neox_response['results'][0]['generated_text'])
            assert neox_text == neox_response['results'][0]['generated_text'], ("generated text from `generate()` and "
                                                                                "`generate_text()` are not the same")

    def test_03_create_mt0_model(self):
        model_id = ModelTypes.MT0_XXL
        if is_cp4d():
            with pytest.raises(WMLClientError):
                _ = Model(model_id=model_id, credentials=self.wml_credentials, project_id=self.project_id)
        else:
            mt0_model = Model(
                model_id=model_id,
                credentials=self.wml_credentials,
                project_id=self.project_id)
            assert mt0_model.get_details()['model_id'] == model_id.value, ("`model_id` from attribute and "
                                                                           "`get_details()` are not the same")

            q = "What is 1 + 1?"

            text_params = {
                GenParams.DECODING_METHOD: DecodingMethods.GREEDY,
                GenParams.MAX_NEW_TOKENS: 10,
                GenParams.STOP_SEQUENCES: ["\n"],
                GenParams.RETURN_OPTIONS: {
                    ReturnOpts.INPUT_TEXT: True,
                    ReturnOpts.GENERATED_TOKENS: True,
                    ReturnOpts.TOP_N_TOKENS: 1
                }
            }
            text = mt0_model.generate_text(prompt=q, params=text_params)
            print(text)
            assert q in text, "generated text do not contain the given prompt"

            response = mt0_model.generate(prompt=q)
            print(response['results'][0]['generated_text'])
            assert response['results'][0]['generated_text'] in text, ("generated text from `generate()` and "
                                                                      "`generate_text()` are not the same")

    def test_04_create_mpt_7b_model(self):
        model_id = 'ibm/mpt-7b-instruct2'
        mpt_7b_params = {
            GenParams.DECODING_METHOD: 'greedy',
            GenParams.MAX_NEW_TOKENS: 1,
            GenParams.MIN_NEW_TOKENS: 0,
        }

        if is_cp4d():
            with pytest.raises(WMLClientError):
                _ = Model(model_id=model_id, params=mpt_7b_params,
                          credentials=self.wml_credentials, project_id=self.project_id)
        else:
            mpt_7b_model = Model(
                model_id=model_id,
                params=mpt_7b_params,
                credentials=self.wml_credentials,
                project_id=self.project_id)
            assert mpt_7b_model.get_details()['model_id'] == model_id, ("`model_id` from attribute and "
                                                                        "`get_details()` are not the same")

            q = "1 + 1?"

            ul2_text = mpt_7b_model.generate_text(prompt=q)
            print(ul2_text)

            ul2_response = mpt_7b_model.generate(prompt=q)
            print(ul2_response['results'][0]['generated_text'])
            assert ul2_text == ul2_response['results'][0]['generated_text'], ("generated text from `generate()` and "
                                                                              "`generate_text()` are not the same")

    def test_05_create_flan_t5_model(self):
        model_id = "google/flan-t5-xxl"
        if is_cp4d():
            with pytest.raises(WMLClientError):
                _ = Model(model_id=model_id, credentials=self.wml_credentials, project_id=self.project_id)
        else:
            flan_t5_model = Model(
                model_id=model_id,
                credentials=self.wml_credentials,
                project_id=self.project_id)
            assert flan_t5_model.get_details()['model_id'] == model_id, ("`model_id` from attribute and "
                                                                         "`get_details()` are not the same")

            q = "What is 1 * 1?"

            text_params = {
                GenParams.DECODING_METHOD: "greedy",
                GenParams.MAX_NEW_TOKENS: 20,
                GenParams.MIN_NEW_TOKENS: 0,
                GenParams.RANDOM_SEED: 99,
                GenParams.STOP_SEQUENCES: ["\n"],
                GenParams.TEMPERATURE: 0,
                GenParams.TIME_LIMIT: 100,
                GenParams.TOP_K: 99,
                GenParams.TOP_P: 0.99,
                GenParams.REPETITION_PENALTY: 1.5,
                GenParams.TRUNCATE_INPUT_TOKENS: 299,
                GenParams.RETURN_OPTIONS: {
                    "input_text": True,
                    "generated_tokens": True,
                    "top_n_tokens": 1
                }
            }
            text = flan_t5_model.generate_text(prompt=q, params=text_params)
            print(text)
            assert q in text, "generated text do not contain the given prompt"

            response = flan_t5_model.generate(prompt=q)
            print(response['results'][0]['generated_text'])
            assert response['results'][0]['generated_text'] in text, ("generated text from `generate()` and "
                                                                      "`generate_text()` are not the same")

    def test_06_create_starcoder_model(self):
        model_id = ModelTypes.STARCODER
        if is_cp4d():
            with pytest.raises(WMLClientError):
                _ = Model(model_id=model_id, credentials=self.wml_credentials, project_id=self.project_id)
        else:
            starcoder = Model(
                model_id=model_id,
                credentials=self.wml_credentials,
                project_id=self.project_id)

            assert starcoder.get_details()['model_id'] in model_id.value, ("`model_id` from attribute and "
                                                                           "`get_details()` are not the same")

            q = """
    Using the directions below, generate Python code for the given task.

    Input:
    # Write a Python function that prints 'Hello World!' string 'n' times.

    Output:
    def print_n_times(n):
        for i in range(n):
            print("Hello World!")

    <end of code>

    Input:
    # Write a Python function, which generates sequence of prime numbers.
    # The function 'primes' will take the argument 'n', an int. It will return a list which contains all primes less than 'n'.

    Output:
            """

            text_params = {
                GenParams.DECODING_METHOD: "greedy",
                GenParams.MAX_NEW_TOKENS: 1000,
                GenParams.MIN_NEW_TOKENS: 1,
                GenParams.STOP_SEQUENCES: ["<end of code>"],
                GenParams.REPETITION_PENALTY: 1,
                GenParams.RETURN_OPTIONS: {
                    "input_text": True,
                    "generated_tokens": True,
                    "top_n_tokens": 1
                }
            }
            text = starcoder.generate_text(prompt=q, params=text_params)
            print("*** GENERATE_TEXT ***")
            print(text)
            assert q in text, "generated text do not contain the given prompt"

            updated_text_params = {
                GenParams.DECODING_METHOD: "greedy",
                GenParams.MAX_NEW_TOKENS: 500,
                GenParams.MIN_NEW_TOKENS: 20,
                GenParams.STOP_SEQUENCES: ["<end of code>"],
                GenParams.REPETITION_PENALTY: 1
            }

            response = starcoder.generate(prompt=q, params=updated_text_params)
            print("*** GENERATE ***")
            print(response['results'][0]['generated_text'])
            assert response['results'][0]['generated_text'] in text, ("generated text from `generate()` and "
                                                                      "`generate_text()` are not the same")

    def test_07_create_llama2_70b_chat_model(self):
        model_id = ModelTypes.LLAMA_2_70B_CHAT
        if is_cp4d():
            with pytest.raises(WMLClientError):
                _ = Model(model_id=model_id, credentials=self.wml_credentials,
                          project_id=self.project_id)
        else:
            llama2 = Model(
                model_id=model_id,
                credentials=self.wml_credentials,
                project_id=self.project_id)
            assert llama2.get_details()['model_id'] == model_id.value, ("`model_id` from attribute and "
                                                                        "`get_details()` are not the same")

            q = "Answer the question. What is 1 * 1?"

            text_params = {
                GenParams.RETURN_OPTIONS: {
                    "input_text": True,
                    "generated_tokens": True,
                    "top_n_tokens": 1
                }
            }
            text = llama2.generate_text(prompt=q, params=text_params)
            print("TEXT", text)
            assert q in text, "generated text do not contain the given prompt"

            response = llama2.generate(prompt=q)
            print("RESPONSE", response['results'][0]['generated_text'])
            assert response['results'][0]['generated_text'] in text, ("generated text from `generate()` and "
                                                                      "`generate_text()` are not the same")

    def test_08_create_granite_13b_instruct_model(self):
        model_id = ModelTypes.GRANITE_13B_INSTRUCT
        if is_cp4d():
            with pytest.raises(WMLClientError):
                _ = Model(model_id=model_id, credentials=self.wml_credentials,
                          project_id=self.project_id)
        else:
            granite_instruct = Model(
                model_id=model_id,
                credentials=self.wml_credentials,
                project_id=self.project_id)
            assert granite_instruct.get_details()['model_id'] == model_id.value, ("`model_id` from attribute and "
                                                                                  "`get_details()` are not the same")

            q = "Answer the question. What is 1 * 1?"

            text_params = {
                GenParams.RETURN_OPTIONS: {
                    "input_text": True,
                    "generated_tokens": True,
                    "top_n_tokens": 1
                }
            }
            text = granite_instruct.generate_text(prompt=q, params=text_params)
            print("TEXT", text)
            assert q in text, "generated text do not contain the given prompt"

            response = granite_instruct.generate(prompt=q)
            print("RESPONSE", response['results'][0]['generated_text'])
            assert response['results'][0]['generated_text'] in text, ("generated text from `generate()` and "
                                                                      "`generate_text()` are not the same")

    def test_09_create_granite_13b_chat_model(self):
        model_id = ModelTypes.GRANITE_13B_CHAT
        granite_chat = Model(
            model_id=model_id,
            credentials=self.wml_credentials,
            project_id=self.project_id)
        assert granite_chat.get_details()['model_id'] == model_id.value, ("`model_id` from attribute and "
                                                                          "`get_details()` are not the same")

        q = "Answer the question. What is 1 * 1?"

        text_params = {
            GenParams.RETURN_OPTIONS: {
                "input_text": True,
                "generated_tokens": True,
                "top_n_tokens": 1
            }
        }
        text = granite_chat.generate_text(prompt=q, params=text_params)
        print("TEXT", text)
        assert q in text, "generated text do not contain the given prompt"

        response = granite_chat.generate(prompt=q)
        print("RESPONSE", response['results'][0]['generated_text'])
        assert response['results'][0]['generated_text'] in text, ("generated text from `generate()` and "
                                                                  "`generate_text()` are not the same")

    def test_10_generate_text_with_raw_response(self):
        model_id = ModelTypes.FLAN_T5_XL
        ul2_params = {
            GenParams.DECODING_METHOD: DecodingMethods.GREEDY,
            GenParams.MAX_NEW_TOKENS: 50,
            GenParams.STOP_SEQUENCES: ['\n\n']
        }

        flan_t5_model = Model(
            model_id=model_id,
            params=ul2_params,
            credentials=self.wml_credentials,
            project_id=self.project_id)

        q = "What is 1 + 1?"

        ul2_raw_generate_text = flan_t5_model.generate_text(prompt=q, raw_response=True)
        ul2_response_generate = flan_t5_model.generate(prompt=q)

        assert ul2_raw_generate_text['results'] == ul2_response_generate['results'], ("Response from `generate()` and "
                                                                                      "`generate_text()` with raw_response=True "
                                                                                      "should be the same")

        sample_q = "What is 1 + {}?"
        prompts = [sample_q.format(i) for i in range(20)]

        ul2_raw_generate_text = flan_t5_model.generate_text(prompt=prompts, concurrency_limit=5, raw_response=True)
        assert len(prompts) == len(ul2_raw_generate_text), \
            "Length of texts list is not equal length of prompts list"

        ul2_response_generate = flan_t5_model.generate(prompt=prompts, concurrency_limit=6)
        assert len(prompts) == len(ul2_response_generate), "Length of responses is not equal length of prompts list"

        for text, response in zip(ul2_raw_generate_text, ul2_response_generate):
            assert text['results'][0]['generated_text'] == response['results'][0]['generated_text'],\
                f"Methods outputs are not equal generate_text: {text}, generate: {response}"

    def test_11_generate_text_stream(self):
        model_id = ModelTypes.LLAMA_2_13B_CHAT
        text_params = {
            GenParams.DECODING_METHOD: DecodingMethods.GREEDY,
            GenParams.MIN_NEW_TOKENS: 0,
            GenParams.MAX_NEW_TOKENS: 20
        }
        model = Model(
            model_id=model_id,
            params=text_params,
            credentials=self.wml_credentials,
            project_id=self.project_id)

        q = "Write an epigram about the sun"
        text = model.generate_text(prompt=q)
        text_stream = model.generate_text_stream(prompt=q)

        linked_text_stream = ''
        for chunk in text_stream:
            assert isinstance(chunk, str), f"chunk expect type '{str}', actual '{type(chunk)}'"
            linked_text_stream += chunk

        assert text == linked_text_stream, "Linked text stream are not the same as generated text"

    def test_12_create_flan_t5_xl_model(self):
        model_id = ModelTypes.FLAN_T5_XL
        flan_t5_xl = Model(
            model_id=model_id,
            credentials=self.wml_credentials,
            project_id=self.project_id)
        assert flan_t5_xl.get_details()['model_id'] == model_id.value, ("`model_id` from attribute and "
                                                                        "`get_details()` are not the same")

        q = "Answer the question. What is 1 * 1?"
        text_params = {
            GenParams.RETURN_OPTIONS: {
                "input_text": True,
                "generated_tokens": True,
                "top_n_tokens": 1
            }
        }
        text = flan_t5_xl.generate_text(prompt=q, params=text_params)
        print("TEXT", text)
        assert q in text, "generated text do not contain the given prompt"

        response = flan_t5_xl.generate(prompt=q)
        print("RESPONSE", response['results'][0]['generated_text'])
        assert response['results'][0]['generated_text'] in text, ("generated text from `generate()` and "
                                                                  "`generate_text()` are not the same")

    def test_13_generate_text_stream_with_raw_response(self):
        raw_response = True
        text_params = {
            GenParams.DECODING_METHOD: DecodingMethods.GREEDY,
            GenParams.MIN_NEW_TOKENS: 0,
            GenParams.MAX_NEW_TOKENS: 20
        }
        model_id = ModelTypes.LLAMA_2_13B_CHAT
        model = Model(
            model_id=model_id,
            params=text_params,
            credentials=self.wml_credentials,
            project_id=self.project_id)

        q = "Write an epigram about the sun"
        text = model.generate_text(prompt=q)
        text_stream = model.generate_text_stream(prompt=q, raw_response=raw_response)

        linked_text_stream = ''
        for ind, chunk in enumerate(text_stream):
            if ind == 0:
                assert isinstance(chunk, dict), f"chunk expect type '{dict}', actual '{type(chunk)}'"
                assert isinstance(chunk["model_id"], str), (f"model_id expect type '{str}', "
                                                            f"actual '{type(chunk['model_id'])}'")
                assert isinstance(chunk["created_at"], str), (f"created_at expect type '{str}', "
                                                              f"actual '{type(chunk['created_at'])}'")
                assert isinstance(chunk["results"], list), (f"results expect type '{list}', "
                                                            f"actual '{type(chunk['results'])}'")

                assert chunk["model_id"] == model_id.value, (f"Wrong model naming, expected `{model_id.value}`, "
                                                             f"actual `{chunk['model_id']}`")

            linked_text_stream += chunk['results'][0]['generated_text']

        assert text == linked_text_stream, "Linked text stream are not the same as generated text"

    def test_13b_generate_async(self):
        model_id = ModelTypes.FLAN_UL2
        text_params = {
            GenParams.DECODING_METHOD: DecodingMethods.GREEDY,
            GenParams.MIN_NEW_TOKENS: 0,
            GenParams.MAX_NEW_TOKENS: 20
        }
        model = Model(
            model_id=model_id,
            params=text_params,
            credentials=self.wml_credentials,
            project_id=self.project_id)

        prompts = ["Write an epigram about the sun", "What is molecule?", "What is orange?"]
        expected_outputs = model.generate_text(prompt=prompts)
        responses = model.generate(prompt=prompts, async_mode=True)

        assert isinstance(responses, Generator), "Method `generate()` with async_mode didn't return generator"

        outputs, inputs = [], []
        for response in responses:
            generated_text = response['results'][0]['generated_text']
            inputs.append(generated_text.split('\n')[0])
            outputs.append(generated_text.split('\n')[-1])

        assert set(inputs) == set(prompts), f"Inputs {inputs} different than prompts {prompts}"
        assert set(outputs) == set(expected_outputs), f"Outputs {outputs} different than expected {expected_outputs}"

    @pytest.mark.parametrize('model_id', available_models)
    def test_14_tokenize_without_return_tokens(self, model_id):

        return_tokens = False

        model = Model(
            model_id=model_id,
            credentials=self.wml_credentials,
            project_id=self.project_id)

        q = "Write an epigram about the sun"
        tokenized_response = model.tokenize(prompt=q, return_tokens=return_tokens)
        assert isinstance(tokenized_response, dict), f"Response type is {type(tokenized_response)}, but should be dict"

        tokenized_response_model_id = tokenized_response.get("model_id")
        assert model_id == tokenized_response_model_id, "`model_id` for POST and response are not the same"

        tokenized_response_token_count = tokenized_response.get("result").get("token_count")
        assert isinstance(tokenized_response_token_count, int), (f"`token_count` type is {type(tokenized_response_token_count)}, "
                                                                 f"but should be int")

        tokenized_response_tokens = tokenized_response.get("result").get("tokens")
        assert not tokenized_response_tokens, "The response received `tokens` when there shouldn't be any"

    @pytest.mark.parametrize('model_id', available_models)
    def test_15_tokenize_with_return_tokens(self, model_id):

        return_tokens = True

        model = Model(
            model_id=model_id,
            credentials=self.wml_credentials,
            project_id=self.project_id)

        q = "Write an epigram about the sun"
        tokenized_response = model.tokenize(prompt=q, return_tokens=return_tokens)
        assert isinstance(tokenized_response, dict), f"Response type is {type(tokenized_response)}, but should be dict"

        tokenized_response_model_id = tokenized_response.get("model_id")
        assert model_id == tokenized_response_model_id, "`model_id` for POST and response are not the same"

        tokenized_response_token_count = tokenized_response.get("result").get("token_count")
        assert isinstance(tokenized_response_token_count, int), (f"`token_count` type is {type(tokenized_response_token_count)}, "
                                                                 f"but should be int")

        tokenized_response_tokens = tokenized_response.get("result").get("tokens")
        assert isinstance(tokenized_response_tokens, list), (f"`tokens` type is {type(tokenized_response_tokens)}, "
                                                             f"but should be list")

        assert tokenized_response_token_count == len(tokenized_response_tokens), (f"The number of tokens in the list does "
                                                                                  f"not equal the number of tokens returned")

    @pytest.mark.parametrize('model_id', available_models)
    def test_16_tokenize_input_token(self, model_id):

        return_tokens = True
        q = "Write an epigram about the sun"

        model_1 = Model(
            model_id=model_id,
            credentials=self.wml_credentials,
            project_id=self.project_id)
        model_1_response = model_1.tokenize(prompt=q, return_tokens=return_tokens)
        model_1_input_tokens = model_1_response.get("result").get("tokens")

        params = {
            GenParams.RETURN_OPTIONS: {
                ReturnOpts.INPUT_TOKENS: True
            }
        }
        model_2 = Model(
            model_id=model_id,
            credentials=self.wml_credentials,
            project_id=self.project_id,
            params=params)
        model_2_response = model_2.generate(prompt=q)
        model_2_input_tokens = [token["text"] for token in model_2_response.get("results")[0].get("input_tokens")]

        assert model_1_input_tokens == model_2_input_tokens, ("Tokens from `tokenize()` and `generate()` with "
                                                              "`input_tokens=True` are not the same")

    def test_17_create_elyza_japanese_llama_2_7b_instruct_model(self):

        model_id = ModelTypes.ELYZA_JAPANESE_LLAMA_2_7B_INSTRUCT
        model_params = {
            GenParams.DECODING_METHOD: DecodingMethods.GREEDY,
            GenParams.MAX_NEW_TOKENS: 15,
        }

        if is_cp4d():
            with pytest.raises(WMLClientError):
                _ = Model(model_id=model_id, params=model_params,
                          credentials=self.wml_credentials, project_id=self.project_id)

        else:
            elyza_model = Model(
                model_id=model_id,
                params=model_params,
                credentials=self.wml_credentials,
                project_id=self.project_id)
            assert elyza_model.get_details()['model_id'] == model_id.value, ("`model_id` from attribute and "
                                                                             "`get_details()` are not the same")

            q = "Translate to japanese: Cat has 4 legs"

            elyza_text = elyza_model.generate_text(prompt=q)
            print(elyza_text)

            elyza_response = elyza_model.generate(prompt=q)
            print(elyza_response['results'][0]['generated_text'])
            assert elyza_text == elyza_response['results'][0]['generated_text'], (
                "generated text from `generate()` and "
                "`generate_text()` are not the same")