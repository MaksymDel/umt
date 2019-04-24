
Train:
`allennlp train fixtures/configs/proto.jsonnet -s fixtures/serialized/proto/ --include-package unsupervised_translation -f`

Evaluate:
`allennlp evaluate fixtures/serialized/proto_fs/model.tar.gz '{"en-ru": "fixtures/data/para.en-ru"}' --include-package unsupervised_translation`

Translate (WIP):
`echo '{"sentence": "Did Uriah honestly think he could beat The Legend of Zelda in under three hours?"}' | \
allennlp predict \
    fixtures/serialized/proto/model.tar.gz --include-package unsupervised_translation - `


