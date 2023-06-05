from .utils import (
    get_tokenizer,
    get_dataset,
    get_args
)
from transformers import AutoTokenizer

def prepare_data():
  args = get_args()
  tokenizer = get_tokenizer(args)
  dataset = get_dataset(args, tokenizer, save_data=True)
  print(dataset)
  return dataset

if __name__ == "__main__":
  dataset = prepare_data()
  #dataset.save_to_disk("/content/sample_data/hf_data")
