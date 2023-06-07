# chat-sentiment-analysis
Solve all sentiment analysis tasks by chat

## Prepare Finetuning Data
- chat_sentiment_analysis/prepare_data/process_asote_data.py
  - supported tasks
    - Aspect Term Extraction (ATE)
    - Opinion Term Extraction (OTE)
    - Aspect Term-Opinion Term Pair Extraction (AOPE)
    - Aspect term, Sentiment, Opinion term Triplet Extraction (ASOTE)
- chat_sentiment_analysis/prepare_data/process_acsa_data.py
  - supported tasks
    - Aspect Category Detection (ACD)
    - Aspect Category-Sentiment Pair Extraction (ACSA)
- chat_sentiment_analysis/prepare_data/process_acos_data.py
  - supported tasks
    - [Aspect-Category-Opinion-Sentiment (ACOS) Quadruple Extraction](https://github.com/NUSTM/ACOS)
- chat_sentiment_analysis/prepare_data/process_structured_sentiment_data.py
  - supported tasks
    - [Holder, Target, Opinion, Sentiment (HTOS) Quadruple Extraction](https://github.com/jerbarnes/semeval22_structured_sentiment)

## Step by Step
- [filetune](chat_sentiment_analysis/llama/finetune.py)
  - nohup sh run.sh chat_sentiment_analysis/llama/finetune.py > finetune.log 2>&1 &
- [inference](chat_sentiment_analysis/llama/inference.py)
  - nohup sh run.sh chat_sentiment_analysis/llama/inference.py > inference.log 2>&1 &

## Supported Tasks
### Aspect Term Extraction
Instruction: extract aspect terms from the sentence
![](./figures/tasks/ATE.png)

### Opinion Term Extraction
Instruction: extract opinion terms from the sentence

![](./figures/tasks/OTE.png)

### Aspect Term-Opinion Term Pair Extraction
Instruction: extract aspect term-opinion term pairs from the sentence
![](./figures/tasks/AOP.png)

### Aspect Term, Sentiment, Opinion Term Triplet Extraction
Instruction: extract aspect term, sentiment, opinion term triplets from the sentence
![](./figures/tasks/ASOTE.png)

### Aspect Category Detection
Instruction: detect aspect categories from the sentence
![](./figures/tasks/ACD.png)

### Aspect Category-Sentiment Pair Prediction
Instruction: detect aspect category-sentiment pairs from the sentence
![](./figures/tasks/ACSA.png)

### Aspect-Category-Opinion-Sentiment (ACOS) Quadruple Extraction
Instruction: extract Aspect-Category-Opinion-Sentiment Quadruple from the sentence
![](./figures/tasks/ACOS.png)

### Holder, Target, Opinion, Sentiment (HTOS) Quadruple Extraction
Instruction: extract Holder-Target-Opinion-Sentiment Quadruple from the sentence
![](./figures/tasks/HTOS.png)

### Try Chat-Sentiment Yourself

### Tips
- LLaMA-7b, GPU RAM 10G

## Reference
- [my-alpaca](https://github.com/l294265421/my-alpaca)
- [stanford_alpaca](https://github.com/tatsu-lab/stanford_alpaca#fine-tuning)
- [alpaca-lora](https://github.com/tloen/alpaca-lora)
