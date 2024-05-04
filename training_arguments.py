from transformers import TrainingArguments

training_args = TrainingArguments(
    output_dir = "output_marc_ja",     #結果の保存フォルダ
    per_device_train_batch_size= 32,   #訓練時のバッチサイズ 
    per_device_eval_batch_size= 32,    #評価時のバッチサイズ
    learning_rate= 2e-5,               #学習率
    lr_scheduler_type= 'linear',       #学習率のスケジューラの種類
    warmup_ratio= 0.1,                 #学習のウォームアップの長さ
    num_train_epochs= 3,               #訓練エポック数
    save_strategy= 'epoch',            #モデルの保存戦略
    logging_strategy= 'epoch',         #ログの保存戦略
    evaluation_strategy= 'epoch',      #評価の頻度
    load_best_model_at_end= True,      #最良のモデルを最後にロードするか
    metric_for_best_model= 'accuracy', #最良のモデルを選択するメトリック
    # fp16= True,                        #混合精度を使用するか
)