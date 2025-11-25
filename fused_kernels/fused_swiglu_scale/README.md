```bash
# dependency
pip install triton==3.4.0 torch pandas matplotlib

# correctness test
pytest -v ./test_swiglu_triton.py 
pytest -v ./test_swiglu_scale_triton.py 

# perf benchmark
python ./bench_all.py
```
