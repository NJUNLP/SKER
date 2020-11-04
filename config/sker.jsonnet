local project_root = std.extVar("PROJECT_ROOT");
local bert_archive = project_root + "/data/pretrained/chinese_wwm_pytorch/chinese_wwm_pytorch.tar.gz";
local bert_pretrained_model = project_root + "/data/pretrained/chinese_wwm_pytorch/vocab.txt";
local idiom_vector_path = project_root + "/data/idiom_vector.txt";
local idiom_graph_path = project_root + "/data/idiom_graph.json";

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
    "type": "gan",
    "idiom_vector_path": idiom_vector_path,
    "idiom_graph_path": idiom_graph_path,
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

  "train_data_path": project_root + "/data/ChID/train_data.txt",
  "validation_data_path": project_root + "/data/ChID/dev_data.txt",

  "iterator": {
    "type": "bucket",
    "sorting_keys": [["content", "num_tokens"]],
    "batch_size": 64
  },
  "trainer": {
      "optimizer": {
          "type": "adam",
          "lr": 2e-5
      },
      "num_epochs": 10,
      "patience": 5,
      "cuda_device": [0],
      "validation_metric": "+accuracy",
      "num_serialized_models_to_keep": 5
  }
}
