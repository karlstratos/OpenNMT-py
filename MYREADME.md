# Sanity check
python preprocess.py -train_src data/same10-src-train.txt -train_tgt data/same10-tgt-train.txt -valid_src data/same10-src-val.txt -valid_tgt data/same10-tgt-val.txt -save_data data/same10 -src_seq_length 1000000 -tgt_seq_length 1000000 -src_vocab_size 1000000 -tgt_vocab_size 1000000 -share_vocab

python train.py -data data/same10 -save_model models/same10 -share_embeddings -share_decoder_embeddings -label_smoothing 0 -batch_size 64 -max_generator_batches 32 -truncated_decoder 100 -valid_steps 10 -layers 2 -rnn_size 200 -word_vec_size 200 -save_checkpoint_steps 10

python translate.py -model models/same10_step_170_ppl_1.06_acc_98.74.pt -src data/same10-src-val.txt -tgt data/same10-tgt-val.txt -verbose -batch_size 64 -max_length 1000000 -share_vocab -ppl


# Sanity check: transformer
python train.py -data data/same10 -save_model models/same10_transformer -share_embeddings -share_decoder_embeddings -label_smoothing 0 -batch_size 64 -max_generator_batches 2 -truncated_decoder 100 -valid_steps 10 -encoder_type transformer  -decoder_type transformer -position_encoding -layers 6 -rnn_size 10 -word_vec_size 10 -transformer_ff 40 -heads 5 -dropout 0.0 -batch_type tokens -normalization tokens   -accum_count 1 -optim adam -adam_beta2 0.998 -decay_method noam -warmup_steps 8000 -learning_rate 20   -max_grad_norm 0 -param_init 0  -param_init_glorot  -label_smoothing 0.1  -save_checkpoint_steps 10 -input_feed 1


# PTB (configurations following PyTorch example)
python preprocess.py -train_src data/ptb/src-train.txt  -train_tgt data/ptb/tgt-train.txt -valid_src data/ptb/src-val.txt -valid_tgt data/ptb/tgt-val.txt -save_data data/ptb -src_seq_length 1000000 -tgt_seq_length 1000000 -src_vocab_size 1000000 -tgt_vocab_size 1000000

python train.py -data data/ptb -save_model models/ptb -share_decoder_embeddings -batch_size 20 -truncated_decoder 35 -layers 2 -rnn_size 1500 -word_vec_size 1500 -dropout 0.65 -gpu_ranks 0 -log logs/log_ptb_2lstm1500_pytorchex_inputfeed.txt

python translate.py -model models/ptb_step_XYZ -src data/ptb/src-test.txt -tgt data/ptb/tgt-test.txt -verbose -batch_size 64 -max_length 1000000 -ppl


# PTB (training version 2: no input feeding)
python train.py -data data/ptb -save_model models/ptb -share_decoder_embeddings -input_feed 0 -batch_size 20 -truncated_decoder 35 -layers 2 -rnn_size 1500 -word_vec_size 1500 -dropout 0.65 -gpu_ranks 0 -log logs/log_ptb_2lstm1500_pytorchex.txt


# PTB: Transformer
python train.py -data data/ptb -save_model models/ptb_transformer -layers 6 -rnn_size 512 -word_vec_size 512 -transformer_ff 2048 -heads 8 -encoder_type transformer -decoder_type transformer -position_encoding -train_steps 200000 -max_generator_batches 2 -dropout 0.1 -batch_size 4096 -batch_type tokens -normalization tokens -accum_count 1 -optim adam -adam_beta2 0.998 -decay_method noam -warmup_steps 8000 -learning_rate 2 -max_grad_norm 0 -param_init 0  -param_init_glorot -label_smoothing 0.1 -valid_steps 10000 -save_checkpoint_steps 10000 -gpu_ranks 0 -truncated_decoder 35 -share_decoder_embeddings -log logs/log_ptb_6tansformer_512in_2048fat_8heads.txt
