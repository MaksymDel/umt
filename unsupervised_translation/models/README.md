Train:
`allennlp train fixtures/configs/proto.jsonnet -s fixtures/serialized/proto --include-package unsupervised_translation -f`

Evaluate:
`allennlp evaluate fixtures/serialized/proto/model.tar.gz '{"en-ru": "fixtures/data/para.en-ru"}' --include-package unsupervised_translation`

Translate (WIP):
`echo '{"sentence": "Did Uriah honestly think he could beat The Legend of Zelda in under three hours?"}' | \
allennlp predict \
    fixtures/serialized/proto/model.tar.gz --include-package unsupervised_translation - `


TODO0: use LSTM (2 layers) similarly to Artexte for direct comparison

~~TODO1: separate all this stuff to different files and stick to the default trainer and config~~

~~TODO2: use jsonnet config to pass dataset reader to both model and train command~~

~~TODO3: during backtranslation, first greedly backtranslate with torch.not_grad()~~

TODO4: then convert output into list of instances into "Batch" into tensor_dict

TODO5: call model forward on this tensor_dict and get gradients

TODO6: then repeat for other languages (for loop), but reuse the same variables to save GPU
memory

TODO7: finally refactor to get ready for transformers

TODO8: use fairseq transformer