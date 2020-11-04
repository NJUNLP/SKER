# local bert_archive = "../data/pretrained/chinese_wwm_pytorch/chinese_wwm_pytorch.tar.gz";
# local bert_pretrained_model = "../data/pretrained/chinese_wwm_pytorch/vocab.txt";
# local idiom_vector_path = "../data/idiom_vector.txt";
local project_root = std.extVar("PROJECT_ROOT");
local bert_archive = project_root + "/data/pretrained/chinese_wwm_pytorch/chinese_wwm_pytorch.tar.gz";
local bert_pretrained_model = project_root + "/data/pretrained/chinese_wwm_pytorch/vocab.txt";
local idiom_vector_path = project_root + "/data/idiom_vector.txt";
local idiom_definition_path = project_root + "/data/idiom_definition.json";

{
  "dataset_reader": {
    "type": "definition_reader",
    "idiom_vector_path": idiom_vector_path,
    "idiom_definition_path": idiom_definition_path,
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
    "type": "definition",
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