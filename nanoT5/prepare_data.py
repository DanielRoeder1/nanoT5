from .utils import (
    get_tokenizer,
    get_dataset,
    get_args
)
from transformers import AutoTokenizer

def prepare_data():
  args = get_args()
  tokenizer = get_tokenizer(args)
  if args.model.knowledge_injection:
    know_tokenizer = AutoTokenizer.from_pretrained(args.model.know_enc_name) if args.model.know_enc_name != "T5" else tokenizer
  dataset = get_dataset(args.data.data_dir, tokenizer, args know_tokenizer=know_tokenizer, save_data=False)
  print(dataset)
  return dataset

if __name__ == "__main__":
  dataset = prepare_data()
  dataset.save_to_disk("/content/sample_data/hf_data")
