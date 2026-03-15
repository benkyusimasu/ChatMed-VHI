.PHONY: install test lint format clean run-bert run-gpt run-metrics

# 安装依赖
install:
	pip install -r requirements.txt

# 运行测试
test:
	python -m pytest tests/ -v

# 代码检查
lint:
	flake8 BERT_fine_tune/ GPT_fine_tune/ 

# 代码格式化
format:
	black BERT_fine_tune/ GPT_fine_tune/

# 清理缓存
clean:
	rm -rf __pycache__ */__pycache__ .pytest_cache *.pyc

# 运行BERT微调
run-bert:
	cd BERT_fine_tune && jupyter notebook Bert_finetune.ipynb

# 运行GPT微调
run-gpt:
	cd GPT_fine_tune && jupyter notebook GPT_fine_tune.ipynb

# 运行评估
run-metrics:
	cd Metrics && jupyter notebook Metrics.ipynb
