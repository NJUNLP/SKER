# local bert_archive = "../data/pretrained/chinese_wwm_pytorch/chinese_wwm_pytorch.tar.gz";
# local bert_pretrained_model = "../data/pretrained/chinese_wwm_pytorch/vocab.txt";
# local idiom_vector_path = "../data/idiom_vector.txt";
local bert_archive = std.extVar("BERT_ARCHIVE");
local bert_pretrained_model = std.extVar("BERT_PRETRAINED_MODEL");
local idiom_vector_path = std.extVar("IDIOM_VECTOR_PATH");
local project_root = std.extVar("PROJECT_ROOT");

{
  "dataset_reader": {
    "type": "baseline_reader",
    "idiom_vector_path": idiom_vector_path,
    "pretrained_model": bert_pretrained_model,
    "content_token_indexer": {
      "bert": {
        "type": "bert-pretrained",
        "pretrained_model": bert_pretrained_model
      }
    },
    "max_seq_length": 256,
    "lazy": true
  },

  "model": {
    "type": "baseline",
    "idiom_vector_path": idiom_vector_path,
    "content_embedder": {
      "allow_unmatched_keys": true,
      "embedder_to_indexer_map": {
        "bert": ["bert", "bert-offsets"]
      },
      "token_embedders": {
        "bert": {
          "type": "bert-pretrained",
          "pretrained_model": bert_archive,
          "top_layer_only": true,
          "requires_grad": false
        }
      }
    },
    "dropout": 0.2
  },

  "train_data_path": project_root + "src/tests/fixtures/data/chid.txt",
  "validation_data_path": self.train_data_path,

  "iterator": {
    "type": "bucket",
    "sorting_keys": [["content", "num_tokens"]],
    "batch_size": 16
  },
  "trainer": {
      "optimizer": {
          "type": "adam",
          "lr": 2e-5
      },
      "num_epochs": 10,
      "patience": 5,
      "cuda_device": -1,
      "validation_metric": "+accuracy"
  }
}