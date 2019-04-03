local Reader() = {
  "type": "unsupervised_translation"
};  

{
  "train_data_path": {"en": "fixtures/data/mono.en", "ru": "fixtures/data/mono.ru", "en-ru": "fixtures/data/para.en-ru"},
  "validation_data_path": {"en-ru": "fixtures/data/para.en-ru"},
  "dataset_reader": Reader(),
    "model": {
      "type": "unsupervised_translation",
      "dataset_reader": Reader(),
      "source_embedder": {
        "token_embedders": {
          "tokens": {
            "type": "embedding",
            "embedding_dim": 300
          }
        }
      },
      // "beam_size": 2,
      "encoder": {
        "type": "lstm",
        "input_size": 300,
        "hidden_size": 300,
        "bidirectional": true
      },
      "max_decoding_steps": 10
    },
    "iterator": {
      "type": "homogeneous_batches",
      "type_field_name": "lang_pair",
      "batch_size": 2
    },
    "trainer": {
      "num_epochs": 130,
      "patience": 10,
      "cuda_device": -1,
      // "grad_clipping": 5.0,
      "validation_metric": "+BLEU",
      "optimizer": {
        "type": "adam",
        "lr": "0.1"
      }
    }
  }