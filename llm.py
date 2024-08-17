from transformers import (
    AutoModelForCausalLM,
    BitsAndBytesConfig,
)
import bitsandbytes as bnb
import torch


class CausalLM:
    repo_id: str
    quantization_config: BitsAndBytesConfig
    attn_implementation: str

    def __init__(
        self,
        repo_id: str,
        quantization_config: str | BitsAndBytesConfig = "default",
        attn_implementation="default",
    ) -> None:
        self.repo_id = repo_id

        if attn_implementation == "default":
            self.attn_implementation = "eager"
        else:
            self.attn_implementation = attn_implementation

        if quantization_config == "default":
            self.quantization_config = self._create_default_quantization_config()
        else:
            self.quantization_config = quantization_config

    @property
    def model(self):
        return self._initialize_model(self.attn_implementation)

    @property
    def linear_layer_names(self):
        return self._find_all_linear_names()

    def config_quantization(
        self,
        load_in_4bit: bool = True,
        bnb_4bit_quant_type: str = "nf4",
        bnb_4bit_use_double_quant: bool = True,
        bnb_4bit_compute_dtype: torch.dtype = torch.float16,
    ):
        self.quantization_config = BitsAndBytesConfig(
            load_in_4bit=load_in_4bit,
            bnb_4bit_quant_type=bnb_4bit_quant_type,
            bnb_4bit_use_double_quant=bnb_4bit_use_double_quant,
            bnb_4bit_compute_dtype=bnb_4bit_compute_dtype,
        )

    def _create_default_quantization_config(self):
        return BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch.float16,
        )

    def _find_all_linear_names(self):
        target_class = bnb.nn.Linear4bit
        linear_layer_names = set()

        for name_list, module in self.model.named_modules():
            if isinstance(module, target_class):
                names = name_list.split(".")
                layer_name = names[-1] if len(names) > 1 else names[0]
                linear_layer_names.add(layer_name)

        if "lm_head" in linear_layer_names:
            linear_layer_names.remove("lm_head")

        return list(linear_layer_names)

    def _initialize_model(self, attn_implementation: str):
        model = AutoModelForCausalLM.from_pretrained(
            pretrained_model_name_or_path=self.repo_id,
            device_map={"": "cuda"},
            quantization_config=self.quantization_config,
            attn_implementation=attn_implementation,
        )
        model.config.use_cache = False
        model.config.pretraining_tp = 1
        
        return model
