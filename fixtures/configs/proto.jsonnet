local langs_list = ["en", "ru"];
local ae_steps = ["en", "ru"];
local bt_steps = ["en", "en"]; # by now we assume to bt to all tgt langs
local para_steps = ["en-ru"];

local Reader() = {
  "type": "unsupervised_translation",
  "langs_list": langs_list,
  "ae_steps": ae_steps,
  "bt_steps": bt_steps,
  "para_steps": para_steps
};

{
  // "train_data_path": {"en": "fixtures/data/mono.en", "ru": "fixtures/data/mono.ru", "en-ru": "fixtures/data/para.en-ru"},
  // "train_data_path": {"en-ru": "fixtures/data/para.en-ru"},
  "train_data_path": "/home/maksym/research/umt/fixtures/data/",
  // "validation_data_path": {"en-ru": "fixtures/data/para.en-ru"},
  // "validation_data_path": {"en-ru": "fixtures/data/dummy.en-ru"},
  "dataset_reader": Reader(),

  "model": {
      "type": "unsupervised_translation",
      "dataset_reader": Reader(),
      "source_embedder": {
        "token_embedders": {
          "tokens": {
            "type": "embedding",
            "embedding_dim": 512
          }
        }
      },
    },

    "iterator": {
      "type": "homogeneous_batches",
      "type_field_name": "lang_pair",
      "batch_size": 2
    },
    "trainer": {
      "num_epochs": 130,
      "patience": 50,
      "cuda_device": -1,
      // "grad_clipping": 5.0,
      "validation_metric": "+BLEU",
      "optimizer": {
        "type": "adam",
        "lr": "0.01"
      }
    }
  }