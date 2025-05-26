import torch
import os
from transformers import (
    AutoModelForCausalLM,
    AutoConfig,
    AutoTokenizer,
    BitsAndBytesConfig,
)
from peft import (
    get_peft_config,
    get_peft_model,
    LoraConfig,
    TaskType,
    PeftModel
)
from ClassificationHead import ClassificationHead

class ClassificationNetLoRA(torch.nn.Module):
    def __init__(self, MODEL_NAME, DO_TEST, APPLY_LORA):
        super(ClassificationNetLoRA, self).__init__()
        token = "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" 
        model_name = MODEL_NAME #'h2oai/h2o-danube-1.8b-chat' #"microsoft/Phi-3-mini-4k-instruct"
        compute_dtype = getattr(torch, "float16")
        
        # Load configuration from a pre-trained model
        config = AutoConfig.from_pretrained(model_name)
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=False,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=compute_dtype,
        )
        self.model_name = MODEL_NAME
        self.APPLY_LORA = APPLY_LORA
        
        # Load pre-trained language model with specific configurations
        if DO_TEST == False:
            self.llm = AutoModelForCausalLM.from_pretrained(
                model_name, 
                token=token,
                trust_remote_code=True,
                device_map="cuda",
                quantization_config=bnb_config,
            )
        else:
            self.llm = AutoModelForCausalLM.from_pretrained(
                model_name, 
                token=token,
                trust_remote_code=True,
                device_map="cuda",
                quantization_config=bnb_config,
            )

        #--------------------apply LoRA to LLM--------------------
        # LoRA config
        peft_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            inference_mode=False,
            r=8,
            lora_alpha=16,
            target_modules='all-linear',
            lora_dropout=0.1
        )
        
        # Replace language model head with an identity function
        self.llm.lm_head = torch.nn.Identity()

        if DO_TEST:
            if APPLY_LORA == True:
                self.llm.load_adapter("SavedAdapters/MyPeftAdapter")
            self.cls_head = ClassificationHead(config.hidden_size)
            self.cls_head.load_state_dict(torch.load('SavedClassificationModels/clshead.pt'))
            self.cls_head.eval()
            return 
        
        # Apply LoRA - this section is not used by test mode
        if APPLY_LORA == True:
            self.llm = get_peft_model(self.llm, peft_config, 'MyPeftAdapter')
            print(self.llm.active_adapter)
        else:
            # Freeze all parameters of the language model backbone
            for name, param in self.llm.named_parameters():
                param.requires_grad = False
            self.cls_head = ClassificationHead(config.hidden_size)

    # forward pass
    def forward(self, input_ids, attention_mask):
        x = self.llm(input_ids, attention_mask).logits  # get last hidden state
        logits = self.cls_head(x)[:, -1, :]  # Apply classification head to the last token's output
        return logits

    def save_peft_adapter(self):
        if self.APPLY_LORA == False:
            self.llm.save_pretrained('BaseModel')
        else:
            self.llm.save_pretrained('SavedAdapters')  # does not save classification head
        torch.save(self.cls_head.state_dict(), os.path.join('SavedClassificationModels', 'clshead.pt'))

