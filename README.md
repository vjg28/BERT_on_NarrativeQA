# BERT on NarrativeQA 
## Steps:
1) Unzip train.json, dev.json, test.json in `narrative_dataset` folder. 
2) In main repo folder, Terminal Command : `mkdir output_models`
3) `cd pretrained_models` and then `wget https://storage.googleapis.com/bert_models/2018_10_18/uncased_L-12_H-768_A-12.zip`
4) Unzip uncased_L-12_H-768_A-12.zip.
5) Terminal code: `python run_narrativeqa.py   --vocab_file=./pretrained_models/uncased_L-12_H-768_A-12/vocab.txt   --bert_config_file=./pretrained_models/uncased_L-12_H-768_A-12/bert_config.json   --init_checkpoint=./pretrained_models/uncased_L-12_H-768_A-12/bert_model.ckpt   --do_train=True   --train_file=./narrative_dataset/train.json   --do_predict=True   --predict_file=./narrative_dataset/dev.json   --train_batch_size=12   --learning_rate=3e-5   --num_train_epochs=2.0   --max_seq_length=384   --doc_stride=128   --output_dir=./output_models`

## Points:
- Need to experiment on `max_seq_length` and `doc_stride`
