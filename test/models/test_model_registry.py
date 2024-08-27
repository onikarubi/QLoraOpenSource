from qlora.models.model_registry import ModelRegistry
from qlora.logging_formatter import get_logger

logger = get_logger(__name__)

def test_get_models():
    target_model_by_gemma = "google/gemma-2-2b-it"
    target_model_by_llama = "meta-llama/Meta-Llama-3-8B-Instruct"
    target_model_by_elyza_default = "elyza/Llama-3-ELYZA-JP-8B"
    target_model_by_elyza_llama2_default = "elyza/ELYZA-japanese-Llama-2-7b"
    target_model_by_elyza_llama2_instruct = "elyza/ELYZA-japanese-Llama-2-7b-instruct"

    registry = ModelRegistry()
    gemma_model = registry.get_model('gemma', variant='default')
    assert gemma_model == target_model_by_gemma
    logger.debug("Gemma model: {}".format(gemma_model))

    llama_model = registry.get_model('llama', variant='default')
    assert llama_model == target_model_by_llama

    elyza_default_model = registry.get_model('elyza', variant='default')
    assert elyza_default_model == target_model_by_elyza_default

    elyza_llama2_default_model = registry.get_model('elyza', version='llama2', variant='default')
    assert elyza_llama2_default_model == target_model_by_elyza_llama2_default

    elyza_llama2_instruct_model = registry.get_model('elyza', version='llama2', variant='instruct')
    assert elyza_llama2_instruct_model == target_model_by_elyza_llama2_instruct

    logger.debug('Llama model: {}'.format(llama_model))
    logger.debug('Elyza default model: {}'.format(elyza_default_model))
    logger.debug('Elyza llama2 default model: {}'.format(elyza_llama2_default_model))
    logger.debug('Elyza llama2 instruct model: {}'.format(elyza_llama2_instruct_model))
